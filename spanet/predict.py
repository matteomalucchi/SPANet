from argparse import ArgumentParser
from typing import Optional
from numpy import ndarray as Array

import h5py
import numpy as np

from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.dataset.types import Evaluation, SpecialKey, Outputs
from spanet.evaluation import evaluate_on_test_dataset, load_model


def localize_assignment(
    dataset: JetReconstructionDataset,
    assignment: Array,
    event_particle: str,
    product_particle: str,
    source: int
) -> Array:
    """ Convert an assignment from the merged index-space back into the index-space of its own input.

    During training all of the reconstructable inputs are concatenated into a single sequence, so the
    targets of an input are offset by the total size of the inputs preceding it. The predictions live in
    that same merged space, which is not the space used by the indices in the input file. This maps them
    back so that, for example, a product assigned to `JetVBF` is an index into the `JetVBF` collection.

    Products without an explicit input in the event file stay in the merged index-space, since there is
    no single collection to map them into.
    """
    if source < 0:
        return assignment

    offset = int(dataset.source_offsets[source])
    size = int(dataset.source_sizes[source])

    valid = assignment >= 0
    inside = (assignment >= offset) & (assignment < offset + size)

    # The model should never assign a vector outside of the declared input, but if the network was
    # trained without `assignment_source_exclusivity` then it may. Keep the merged indices in that case
    # rather than silently emitting an index into the wrong collection.
    if not (inside | ~valid).all():
        input_name = dataset.event_info.input_names[source]
        print(
            f"Warning: {event_particle}/{product_particle} is assigned to input '{input_name}' but "
            f"{int((valid & ~inside).sum())} predictions fall outside of it. "
            f"Writing merged indices for this product. "
            f"Enable `assignment_source_exclusivity` and retrain to keep the inputs exclusive."
        )
        return assignment

    return np.where(valid, assignment - offset, assignment)


def create_hdf5_output(
    output_file: str,
    dataset: JetReconstructionDataset,
    evaluation: Evaluation,
    full_outputs: Optional[Outputs],
    global_indices: bool = False
):
    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        # Copy over the source features from the input file.
        with h5py.File(dataset.data_file, 'r') as input_dataset:
            for input_name in input_dataset[SpecialKey.Inputs]:
                for feature_name in input_dataset[SpecialKey.Inputs][input_name]:
                    output.create_dataset(
                        f"{SpecialKey.Inputs}/{input_name}/{feature_name}",
                        data=input_dataset[SpecialKey.Inputs][input_name][feature_name]
                    )

        # Construct the assignment structure. Output both the top assignment and associated probabilities.
        for event_particle in dataset.event_info.event_particles:
            product_particles = dataset.event_info.product_particles[event_particle]
            product_sources = product_particles.sources

            for i, product_particle in enumerate(product_particles):
                assignment = evaluation.assignments[event_particle][:, i]
                source = -1 if global_indices else product_sources[i]

                assignment = localize_assignment(
                    dataset, assignment, event_particle, product_particle, source
                )

                product_dataset = output.create_dataset(
                    f"{SpecialKey.Targets}/{event_particle}/{product_particle}",
                    data=assignment
                )

                # Record which input collection these indices refer to so the output is unambiguous.
                if product_sources[i] >= 0:
                    product_dataset.attrs["input"] = dataset.event_info.input_names[product_sources[i]]
                    product_dataset.attrs["merged_indices"] = source < 0

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/assignment_probability",
                data=evaluation.assignment_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/detection_probability",
                data=evaluation.detection_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/marginal_probability",
                data=(
                    evaluation.detection_probabilities[event_particle] *
                    evaluation.assignment_probabilities[event_particle]
                )
            )

        # Simply copy over the structure of the regressions and classifications.
        for name, regression in evaluation.regressions.items():
            output.create_dataset(f"{SpecialKey.Regressions}/{name}", data=regression)

        for name, classification in evaluation.classifications.items():
            output.create_dataset(f"{SpecialKey.Classifications}/{name}", data=classification)

        if full_outputs is not None:
            for name, vector in full_outputs.vectors.items():
                output.create_dataset(f"{SpecialKey.Embeddings}/{name}", data=vector)


def main(log_directory: str,
         output_file: str,
         checkpoint: str,
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         output_vectors: bool,
         global_indices: bool,
         gpu: bool,
         fp16: bool):
    model = load_model(log_directory, test_file, event_file, batch_size, gpu, fp16=fp16, checkpoint=checkpoint)

    if output_vectors:
        evaluation, full_outputs = evaluate_on_test_dataset(model, return_full_output=True, fp16=fp16)
    else:
        evaluation = evaluate_on_test_dataset(model, return_full_output=False, fp16=fp16)
        full_outputs = None

    create_hdf5_output(output_file, model.testing_dataset, evaluation, full_outputs, global_indices)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("output_file", type=str,
                        help="The output HDF5 to create with the new predicted jets for each event.")

    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None,
                        help="Specify which checkpoint in the log_directory you want to load.")

    parser.add_argument("-tf", "--test_file", type=str, default=None,
                        help="Replace the test file in the options with a custom one. "
                             "Must provide if options does not define a test file.")

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Replace the event file in the options with a custom event.")

    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Replace the batch size in the options with a custom size.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")
    
    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Automatic Mixed Precision for inference.")

    parser.add_argument("-gi", "--global_indices", action="store_true",
                        help="Output the assignment indices in the merged index-space spanning every input "
                             "instead of the index-space of the input each product is assigned to.")

    parser.add_argument("-v", "--output_vectors", action="store_true",
                        help="Include embedding vectors in output in an additional section of the HDF5.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)
