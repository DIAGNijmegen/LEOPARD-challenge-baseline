import os
import json
import torch
import argparse

from pathlib import Path
from datetime import timedelta

from source.dist_utils import is_main_process
from source.model import MIL
from source.components import HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Local HViT", add_help=add_help)
    parser.add_argument("--region-size", default=2048, type=int, help="context size")
    parser.add_argument("--features-dim", default=1024, type=int, help="tile-level features dimension")
    parser.add_argument("--mixed-precision", default=False, type=bool, help="enable mixed precision")
    parser.add_argument("--restrict", default=None, type=str, help="path to a .txt file with ids of cases we want to restrict inference on")
    return parser


def run(args):

    distributed = torch.cuda.device_count() > 1
    if distributed:
        timeout = timedelta(hours=10)
        torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        if is_main_process():
            print(f"Distributed session successfully initialized")
    if is_main_process():
        _show_torch_cuda_info()
        # print contents of input folder
        print("input folder contents:")
        print_directory_contents(INPUT_PATH)
        print("=+=" * 10)

    # set baseline parameters
    region_size = args.region_size
    features_dim = args.features_dim
    mixed_precision = args.mixed_precision
    nbins = 4

    # (optionally) restrict inference
    restrict_ids = None
    if args.restrict is not None:
        restrict_file = Path(args.restrict)
        assert restrict_file.is_file()
        with open(restrict_file, "r") as f:
            restrict_ids = [x.strip() for x in f.readlines()]

    # instantiate feature aggregator
    feature_aggregator_weights = Path(RESOURCE_PATH, f"feature_aggregator.pt")
    feature_aggregator = HierarchicalViT(
        feature_aggregator_weights,
        num_classes=nbins,
        region_size=region_size,
        input_embed_dim=features_dim,
    )
    if is_main_process():
        print("=+=" * 10)

    # instantiate the algorithm
    algorithm = MIL(
        Path(INPUT_PATH, "features"),
        feature_aggregator,
        mixed_precision=mixed_precision,
        restrict_ids=restrict_ids,
        distributed=distributed,
    )

    # forward pass
    predictions = algorithm.process()
    if is_main_process():
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

    args = get_args_parser(add_help=True).parse_args()
    raise SystemExit(run(args))