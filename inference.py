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
import argparse

from pathlib import Path

from source.model import MIL
from source.components import UNI, HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/resources")


def run(args):

    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    # set number of bins
    nbins = 4

    # instantiate feature extractor
    feature_extractor_weights = Path(RESOURCE_PATH, f"feature_extractor.pt")
    feature_extractor = UNI(feature_extractor_weights)
    print("=+=" * 10)

    # instantiate feature aggregator
    feature_aggregator_weights = Path(RESOURCE_PATH, f"feature_aggregator_{args.region_size}_{args.fold}.pt")
    feature_aggregator = HierarchicalViT(
        feature_aggregator_weights,
        num_classes=nbins,
        region_size=args.region_size,
        input_embed_dim=args.features_dim,
    )
    print("=+=" * 10)

    # instantiate the algorithm
    algorithm = MIL(
        feature_extractor,
        feature_aggregator,
        spacing=0.5,
        region_size=args.region_size,
        features_dim=args.features_dim,
        backend="asap",
        batch_size=1,
        num_workers=1,
        nfeats_max=args.nfeats_max,
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--region_size", type=int)
    parser.add_argument("--features_dim", type=int, default=384)
    parser.add_argument("--nfeats_max", type=int)
    args = parser.parse_args()

    raise SystemExit(run(args))