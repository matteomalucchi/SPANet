from typing import Tuple, List, Optional
from opt_einsum import contract_expression

import torch
from torch import Tensor, nn, jit

from spanet.options import Options
from spanet.dataset.types import Symmetries
from spanet.network.utilities import masked_log_softmax
from spanet.network.layers.stacked_encoder import StackedEncoder
from spanet.network.layers.branch_linear import BranchLinear
from spanet.network.symmetric_attention import SymmetricAttentionSplit, SymmetricAttentionFull


class BranchDecoder(nn.Module):
    # noinspection SpellCheckingInspection
    WEIGHTS_INDEX_NAMES = "ijklmn"
    DEFAULT_JET_COUNT = 16

    def __init__(
        self,
        options: Options,
        particle_name: str,
        product_names: List[str],
        product_symmetries: Symmetries,
        product_sources: Optional[List[int]] = None,
        softmax_output: bool = True
    ):
        super(BranchDecoder, self).__init__()

        self.degree = product_symmetries.degree
        self.particle_name = particle_name
        self.product_names = product_names
        self.softmax_output = softmax_output
        self.combinatorial_scale = options.combinatorial_scale

        # The input which every product of this particle is restricted to, -1 meaning any input.
        if product_sources is None:
            product_sources = [-1] * self.degree

        self.product_sources = list(product_sources)
        self.source_exclusivity = (
            options.assignment_source_exclusivity and
            any(source >= 0 for source in self.product_sources)
        )

        # Each branch has a personal encoder stack to extract particle-level data
        self.encoder = StackedEncoder(
            options,
            options.num_branch_embedding_layers,
            options.num_branch_encoder_layers
        )

        # Symmetric attention to create the output distribution
        attention_layer = SymmetricAttentionSplit if options.split_symmetric_attention else SymmetricAttentionFull
        self.attention = attention_layer(options, self.degree, product_symmetries.permutations)

        # Optional output predicting if the particle was present or not
        self.detection_classifier = BranchLinear(options, options.num_detector_layers)

        self.num_targets = len(self.attention.permutation_group)
        self.permutation_indices = self.attention.permutation_indices

        self.padding_mask_operation = self.create_padding_mask_operation(options.batch_size)
        self.diagonal_mask_operation = self.create_diagonal_mask_operation()
        self.source_mask_operation = self.create_source_mask_operation()
        self.diagonal_masks = {}
        self.source_masks = {}

    def create_padding_mask_operation(self, batch_size: int):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->b{weights_index_names}"
        return expression

    def create_diagonal_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->{weights_index_names}"
        return expression

    def create_source_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(weights_index_names)
        expression = f"{operands}->{weights_index_names}"
        return expression

    def create_source_mask(self, output: Tensor, input_index: Tensor) -> Tensor:
        """ Mask forbidding a product from being assigned to a vector of a different input collection.

        Parameters
        ----------
        output : [B, T, T, ...]
            The raw assignment logits, only used for the dtype and the device.
        input_index : [T]
            The index of the input collection which produced each assignable vector.

        Returns
        -------
        source_mask : [1, T, T, ...]
            Positive mask indicating that a combination of vectors respects the input assignments.
        """
        num_jets = input_index.shape[0]

        try:
            return self.source_masks[(num_jets, output.device)]
        except KeyError:
            pass

        # One [T] indicator per product, selecting the vectors which that product is allowed to take.
        source_mask_operands = [
            (input_index == source).type_as(output) if source >= 0 else torch.ones_like(input_index).type_as(output)
            for source in self.product_sources
        ]

        # Combine the individual products into the full assignment tensor.
        source_mask = torch.einsum(self.source_mask_operation, *source_mask_operands)
        source_mask = (source_mask > 0).unsqueeze(0)

        self.source_masks[(num_jets, output.device)] = source_mask
        return source_mask

    def create_output_mask(self, output: Tensor, sequence_mask: Tensor, input_index: Tensor) -> Tensor:
        num_jets = output.shape[1]

        # batch_sequence_mask: [B, T, 1] Positive mask indicating jet is real.
        batch_sequence_mask = sequence_mask.transpose(0, 1).contiguous()

        # =========================================================================================
        # Padding mask
        # =========================================================================================
        padding_mask_operands = [batch_sequence_mask.squeeze(-1) * 1] * self.degree
        padding_mask = torch.einsum(self.padding_mask_operation, *padding_mask_operands)
        padding_mask = padding_mask.bool()

        # =========================================================================================
        # Diagonal mask
        # =========================================================================================
        try:
            diagonal_mask = self.diagonal_masks[(num_jets, output.device)]
        except KeyError:
            identity = 1 - torch.eye(num_jets)
            identity = identity.type_as(output)

            diagonal_mask_operands = [identity * 1] * self.degree
            diagonal_mask = torch.einsum(self.diagonal_mask_operation, *diagonal_mask_operands)
            diagonal_mask = diagonal_mask.unsqueeze(0) < (num_jets + 1 - self.degree)
            self.diagonal_masks[(num_jets, output.device)] = diagonal_mask

        # =========================================================================================
        # Input collection mask
        # =========================================================================================
        output_mask = padding_mask & diagonal_mask

        if self.source_exclusivity:
            output_mask = output_mask & self.create_source_mask(output, input_index)

        return output_mask.bool()

    def forward(
            self,
            event_vectors: Tensor,
            padding_mask: Tensor,
            sequence_mask: Tensor,
            global_mask: Tensor,
            input_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ Create a distribution over jets for a given particle and a probability of its existence.

        Parameters
        ----------
        event_vectors : [T, B, D]
            Hidden activations after central encoder.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.
        global_mask : [T]
            Positive mask indicating that a vector is a sequential, and therefore assignable, vector.
        input_index : [T]
            The index of the input collection which produced each vector.

        Returns
        -------
        selection : [TS, TS, ...]
            Distribution over sequential vectors for the target vectors.
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------
        # Apply the branch's independent encoder to each vector.
        # particle_vectors : [T, B, D]
        # ------------------------------------------------------
        encoded_vectors, particle_vector = self.encoder(event_vectors, padding_mask, sequence_mask)

        # -----------------------------------------------
        # Run the encoded vectors through the classifier.
        # detection: [B, 1]
        # -----------------------------------------------
        detection = self.detection_classifier(particle_vector).squeeze(-1)

        # --------------------------------------------------------
        # Extract sequential vectors only for the assignment step.
        # sequential_particle_vectors : [TS, B, D]
        # sequential_padding_mask : [B, TS]
        # sequential_sequence_mask : [TS, B, 1]
        # --------------------------------------------------------
        sequential_particle_vectors = encoded_vectors[global_mask].contiguous()
        sequential_padding_mask = padding_mask[:, global_mask].contiguous()
        sequential_sequence_mask = sequence_mask[global_mask].contiguous()
        sequential_input_index = input_index[global_mask].contiguous()

        # --------------------------------------------------------------------
        # Create the vector distribution logits and the correctly shaped mask.
        # assignment : [TS, TS, ...]
        # assignment_mask : [TS, TS, ...]
        # --------------------------------------------------------------------
        assignment, daughter_vectors = self.attention(
            sequential_particle_vectors,
            sequential_padding_mask,
            sequential_sequence_mask
        )

        assignment_mask = self.create_output_mask(assignment, sequential_sequence_mask, sequential_input_index)

        # ---------------------------------------------------------------------------
        # Need to reshape output to make softmax-calculation easier.
        # We transform the mask and output into a flat representation.
        # Afterwards, we apply a masked log-softmax to create the final distribution.
        # output : [TS, TS, ...]
        # mask : [TS, TS, ...]
        # ---------------------------------------------------------------------------
        if self.softmax_output:
            original_shape = assignment.shape
            batch_size = original_shape[0]

            assignment = assignment.reshape(batch_size, -1)
            assignment_mask = assignment_mask.reshape(batch_size, -1)

            assignment = masked_log_softmax(assignment, assignment_mask)
            assignment = assignment.view(*original_shape)

            # mask = mask.view(*original_shape)
            # offset = torch.log(mask.sum((1, 2, 3), keepdims=True).float()) * self.combinatorial_scale
            # output = output + offset

        return assignment, detection, assignment_mask, particle_vector, daughter_vectors
