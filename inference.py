import os
import json
import tqdm
import torch
import argparse
import torch.distributed as dist

from pathlib import Path
from datetime import timedelta

from source.dist_utils import is_main_process, is_dist_avail_and_initialized
from source.utils import load_inputs, extract_coordinates, save_patches
from source.model import MIL
from source.components import FM, HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Local HViT", add_help=add_help)
    parser.add_argument("--spacing", default=0.5, type=float, help="pixel spacing in mpp")
    parser.add_argument("--region_size", default=2048, type=int, help="context size")
    parser.add_argument("--fm", default="uni", type=str, help="name of FM to use as tile encoder")
    parser.add_argument("--features_dim", default=1024, type=int, help="tile-level features dimension")
    parser.add_argument("--nregion_max", default=None, type=int, help="maximum number of regions to keep")
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
    spacing = args.spacing
    region_size = args.region_size
    fm = args.fm
    features_dim = args.features_dim
    nbins = 4
    nregion_max = args.nregion_max
    num_workers_data_loading = 4
    num_workers_preprocessing = 4
    batch_size = 4

    # preprocess input
    if is_main_process():
        case_list, mask_list = load_inputs()
        with tqdm.tqdm(
            zip(case_list, mask_list),
            desc="Extracting patch coordinates",
            unit=" case",
            total=len(case_list),
            leave=True,
        ) as t:
            for wsi_fp, mask_fp in t:
                tqdm.tqdm.write(f"Preprocessing {wsi_fp.stem}")
                coord, tissue_pct, level, factor = extract_coordinates(wsi_fp, mask_fp, spacing, region_size, num_workers=num_workers_preprocessing)
                save_patches(wsi_fp, coord, tissue_pct, level, region_size, factor, backend="asap", nregion_max=nregion_max, num_workers=num_workers_preprocessing)
        print("=+=" * 10)

    # wait for all processes to finish preprocessing
    if distributed and is_dist_avail_and_initialized():
        dist.barrier()

    # instantiate feature extractor
    feature_extractor_weights = Path(RESOURCE_PATH, f"feature_extractor.pt")
    feature_extractor = FM(fm, feature_extractor_weights)
    if is_main_process():
        print("=+=" * 10)

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
        feature_extractor,
        feature_aggregator,
        spacing=spacing,
        region_size=region_size,
        features_dim=features_dim,
        backend="asap",
        batch_size=batch_size,
        num_workers_data_loading=num_workers_data_loading,
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