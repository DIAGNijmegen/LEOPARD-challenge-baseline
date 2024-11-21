import os
import json
import tqdm
import torch
import argparse
import torch.distributed as dist

from pathlib import Path
from datetime import timedelta

from source.dist_utils import is_main_process, is_dist_avail_and_initialized
from source.utils import load_inputs, extract_coordinates, save_coordinates, save_patches
from source.model import MIL
from source.components import UNI, Kaiko, HierarchicalViT

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Local HViT", add_help=add_help)
    parser.add_argument("--spacing", default=0.5, type=float, help="pixel spacing in mpp")
    parser.add_argument("--region-size", default=2048, type=int, help="context size")
    parser.add_argument("--fm", default="uni", type=str, help="name of FM to use as tile encoder")
    parser.add_argument("--features-dim", default=1024, type=int, help="tile-level features dimension")
    parser.add_argument("--nregion-max", default=None, type=int, help="maximum number of regions to keep")
    parser.add_argument("--nbins", default=4, type=int, help="number of bins the aggregator was trained for")
    parser.add_argument("--mixed-precision", action="store_true", help="turn on mixed precision during inference")
    parser.add_argument("--save-patches-to-disk", action="store_true", help="save patches to disk as jpg")
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
    nbins = args.nbins
    nregion_max = args.nregion_max
    num_workers_data_loading = 4
    num_workers_preprocessing = 4
    batch_size = 1
    mixed_precision = args.mixed_precision
    save_patches_to_disk = args.save_patches_to_disk

    # create output directories
    coordinates_dir = Path("/tmp/coordinates")
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = None
    if save_patches_to_disk:
        patch_dir = Path("/tmp/patches")
        patch_dir.mkdir(parents=True, exist_ok=True)

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
                coordinates, tissue_pct, level, resize_factor = extract_coordinates(wsi_fp, mask_fp, spacing, region_size, num_workers=num_workers_preprocessing)
                save_coordinates(wsi_fp, coordinates, level, region_size, resize_factor, coordinates_dir)
                if save_patches_to_disk:
                    save_patches(wsi_fp, coordinates, tissue_pct, level, region_size, resize_factor, patch_dir, backend="asap", nregion_max=nregion_max, num_workers=num_workers_preprocessing)
        print("=+=" * 10)

    # wait for all processes to finish preprocessing
    if distributed and is_dist_avail_and_initialized():
        dist.barrier()

    # instantiate feature extractor
    feature_extractor_weights = Path(RESOURCE_PATH, f"feature_extractor.pt")
    if fm == "uni":
        feature_extractor = UNI(feature_extractor_weights)
    elif fm == "kaiko":
        feature_extractor = Kaiko(feature_extractor_weights)
    else:
        raise ValueError(f"Foundation model {fm} not recognized")
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
        coordinates_dir=coordinates_dir,
        patch_dir=patch_dir,
        backend="asap",
        batch_size=batch_size,
        mixed_precision=mixed_precision,
        load_patches_from_disk=save_patches_to_disk,
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