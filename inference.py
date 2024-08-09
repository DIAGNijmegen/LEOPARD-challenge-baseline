import os
import json
import torch

from pathlib import Path

from source.model import MIL
from source.components import UNI, HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def run():

    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    # set baseline parameters
    region_size = 2048
    features_dim = 1024
    nbins = 4
    nfeats_max = None

    # instantiate feature extractor
    feature_extractor_weights = Path(RESOURCE_PATH, f"feature_extractor.pt")
    feature_extractor = UNI(feature_extractor_weights)
    print("=+=" * 10)

    # instantiate feature aggregator
    feature_aggregator_weights = Path(RESOURCE_PATH, f"feature_aggregator.pt")
    feature_aggregator = HierarchicalViT(
        feature_aggregator_weights,
        num_classes=nbins,
        region_size=region_size,
        input_embed_dim=features_dim,
    )
    print("=+=" * 10)

    # instantiate the algorithm
    algorithm = MIL(
        feature_extractor,
        feature_aggregator,
        spacing=0.5,
        region_size=region_size,
        features_dim=features_dim,
        backend="asap",
        batch_size=1,
        num_workers_data_loading=1,
        num_workers_preprocessing=4,
        nfeats_max=nfeats_max,
    )

    # forward pass
    predictions = algorithm.process()
    print("=+=" * 10)

    # save output
    write_json_file(
        location=OUTPUT_PATH / "overall-survival-years.json",
        content=predictions[0]
    )

    # print contents of output folder
    print("output folder contents:")
    print_directory_contents(OUTPUT_PATH)

    return 0


def write_json_file(*, location, content):
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


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
        print(f"- number of devices: {torch.cuda.device_count()}")
        print(
            f"- current device: { (current_device := torch.cuda.current_device())}")
        print(
            f"- properties: {torch.cuda.get_device_properties(current_device).name}")
    print("=+=" * 10)


if __name__ == "__main__":

    raise SystemExit(run())