"""
The following is a the inference code for running the baseline algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import os
import torch

from pathlib import Path

from source.model import HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():

    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    # instantiate the algorithm
    feature_extractor_weights = Path(RESOURCE_PATH, "feature_extractor.pt")
    feature_aggregator_weights = Path(RESOURCE_PATH, "feature_aggregator.pt")
    algorithm = HierarchicalViT(
        feature_extractor_weights,
        feature_aggregator_weights,
        spacing=0.5,
        region_size=2048,
        backend="asap",
    )

    # forward pass
    predictions = algorithm.process()
    print("=+=" * 10)
    print(predictions.head())
    print("=+=" * 10)

    # print contents of output folder
    print("output folder contents:")
    print_directory_contents(OUTPUT_PATH)

    return 0


def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print_directory_contents(child_path)
        else:
            print(child_path)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device).name}")
    print("=+=" * 10)


if __name__ == "__main__":

    raise SystemExit(run())