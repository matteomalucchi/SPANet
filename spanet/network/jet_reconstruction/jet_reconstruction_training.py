from typing import Tuple, Dict, List
import inspect

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.types import SpecialKey, Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence

import mdmm

def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')

        return torch.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[Tensor], detections: List[Tensor], targets):
        symmetric_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor.cpu().numpy():

            # Find the assignment loss for each particle in this permutation.
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask)
                for assignment, detection, (target, mask)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            symmetric_losses.append(torch.stack(current_permutation_loss))

        # Shape: (NUM_PERMUTATIONS, NUM_PARTICLES, 2, BATCH_SIZE)
        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        # combined_loss, _ = symmetric_losses.min(0)
        total_symmetric_loss = symmetric_losses.sum((1, 2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        # Simple average of all losses as a baseline.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def get_kl_loss(
            self,
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return []

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            caller_function = inspect.currentframe().f_back.f_code.co_name
            if caller_function == "training_step":
                self.log("loss/symmetric_loss", kl_loss, sync_dist=True)
            elif caller_function == "compute_validation_losses":
                self.log("validation_loss/symmetric_loss", kl_loss, sync_dist=True)
            else:
                raise ValueError(f"Unknown caller function: {caller_function}")
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return [self.options.kl_loss_scale * kl_loss]

    def get_regression_loss(
            self,
            predictions: Dict[str, Tensor],
            targets:  Dict[str, Tensor]
    ) -> List[Tensor]:
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                caller_function = inspect.currentframe().f_back.f_code.co_name
                if caller_function == "training_step":
                    self.log(f"loss/regression/{key}", current_loss, sync_dist=True)
                elif caller_function == "compute_validation_losses":
                    self.log(f"validation_loss/regression/{key}", current_loss, sync_dist=True)
                else:
                    raise ValueError(f"Unknown caller function: {caller_function}")

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return regression_terms

    def get_classification_loss(
            self,
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            event_weights: Tensor = None,
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            # print("targets",targets)
            # print("targets keys",targets.keys())

            # print("all targets",len(targets[key]))
            # print("signal targets",len(targets[key][targets[key]==1]))
            # print("bckg targets",len(targets[key][targets[key]==0]))

            weight = None if not self.balance_classifications else self.classification_weights[key]

            # print("weights", weight)



            # signal_class_weights= weight[[targets[key]==1]]
            # print("signal class weights", signal_class_weights)
            # bckg_class_weights=weight[targets[key]==0]
            # print("bckg class weights", bckg_class_weights)

            # print("event weights", len(event_weights))
            # print("sum_evenet_weights",sum(event_weights))

            # signal_event_weights= event_weights[targets[key]==1]
            # print("signal event weights",signal_event_weights)
            # bckg_event_weights=event_weights[targets[key]==0]
            # print("bckg event weights", bckg_event_weights)

            # print("signal class weight", weight[1])
            # print("signal event weight", signal_event_weights)
            # print("signal weight", signal_event_weights * weight[1])

            # print("bckg class weight", weight[0])
            # print("bckg event wieght", bckg_event_weights [:2])
            # total_weight_bckg= bckg_event_weights * weight[0]
            # print("total bckg weight", total_weight_bckg [:2])


            if self.balance_events:
                assert event_weights is not None, "Event weights are required for balancing classifications."
                current_loss = F.cross_entropy(
                    current_prediction,
                    current_target,
                    ignore_index=-1,
                    weight=weight,
                    reduction='none'
                )
                # Compute the weighted average of the loss
                current_loss = sum(current_loss * event_weights / sum(event_weights))
            else:
                current_loss = F.cross_entropy(
                    current_prediction,
                    current_target,
                    ignore_index=-1,
                    weight=weight
                )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                caller_function = inspect.currentframe().f_back.f_code.co_name
                if caller_function == "training_step":
                    self.log(f"loss/classification/{key}", current_loss, sync_dist=True)
                elif caller_function == "compute_validation_losses":
                    self.log(f"validation_loss/classification/{key}", current_loss, sync_dist=True)
                else:
                    raise ValueError(f"Unknown caller function: {caller_function}")

        return classification_terms

    def get_assignment_loss(self, assignment_loss):
        return assignment_loss.sum()

    def get_detection_loss(self, detection_loss):
        return detection_loss.sum()

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        outputs = self.forward(batch.sources)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        # Take the weighted average of the symmetric loss terms.
        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        # ===================================================================================================
        # Some basic logging
        # ---------------------------------------------------------------------------------------------------
        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Start constructing the list of all computed loss terms.
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            kl_loss = self.get_kl_loss(outputs.assignments, masks, weights)
            total_loss += kl_loss

        if self.options.regression_loss_scale > 0:
            regression_loss = self.add_regression_loss(outputs.regressions, batch.regression_targets)
            total_loss += regression_loss

        if self.options.classification_loss_scale > 0:
            classification_loss = self.get_classification_loss(outputs.classifications, batch.classification_targets, batch.event_weights)
            total_loss += classification_loss

        if self.options.mdmm_loss_scale > 0:
            if self.options.assignment_loss_scale <= 0:
                raise ValueError("MDMM loss requires assignment loss to be enabled.")
            if self.options.classification_loss_scale < 0:
                raise ValueError("MDMM loss requires classification loss to be enabled.")

            total_loss_before_mdmm = torch.cat([loss.view(-1) for loss in total_loss]).mean()

            # Arguments to pass to the loss functions in the MDMM module
            args_mdmm = [(assignment_loss)]
            if self.options.detection_loss_scale > 0:
                args_mdmm.append((detection_loss))

            mdmm_return = self.mdmm_module(
                total_loss_before_mdmm,
                args_mdmm
            )
            total_loss_after_mdmm = mdmm_return.value
            total_loss.append(assignment_loss)
            total_loss = torch.cat([loss.view(-1) for loss in total_loss]).mean()
            self.log("loss/total_loss", total_loss_after_mdmm, sync_dist=True)
            self.log("loss/total_loss_no_mdmm", total_loss, sync_dist=True)
            return total_loss_after_mdmm
        else:
            if self.options.assignment_loss_scale > 0:
                total_loss.append(assignment_loss)

            # ===================================================================================================
            # Combine and return the loss
            # ---------------------------------------------------------------------------------------------------
            total_loss = torch.cat([loss.view(-1) for loss in total_loss])

            self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

            return total_loss.mean()
