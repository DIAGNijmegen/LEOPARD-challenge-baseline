import os
import tqdm
import wholeslidedata as wsd
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import Optional

from source.wsi import WholeSlideImage


def sort_coords_with_tissue(coords, tissue_percentages):
    # mock region filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in coords]
    # combine mocked filenames with coordinates and tissue percentages
    combined = list(zip(mocked_filenames, coords, tissue_percentages))
    # sort combined list by mocked filenames
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coords = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coords, sorted_tissue_percentages


def keep_top_regions(coordinates, tissue_pct, nregion_max: Optional[int] = None):
    # pair region coordinates with their corresponding tissue percentages and keep track of original indices
    indexed_combined = list(enumerate(zip(coordinates, tissue_pct)))
    # sort by tissue percentage in descending order to find the top nregion_max values, but keep original indices
    if nregion_max and len(indexed_combined) > nregion_max:
        top_indices = sorted(indexed_combined, key=lambda x: x[1][1], reverse=True)[:nregion_max]
    else:
        top_indices = sorted(indexed_combined, key=lambda x: x[1][1], reverse=True)
    # get the original indices of the top nregion_max percentages
    top_indices = {i for i, _ in top_indices}
    # filter the original list to keep only the top M regions
    top_coordinates = [coord for i, coord in enumerate(coordinates) if i in top_indices]
    return top_coordinates


def load_inputs():
    """
    Read from /input/
    """
    case_list = sorted([fp for fp in Path("/input/images/prostatectomy-wsi").glob("*.tif")])
    mask_list = sorted([fp for fp in Path("/input/images/prostatectomy-tissue-mask").glob("*.tif")])
    # case_dict = {fp.stem: fp for fp in case_list}
    # mask_dict = {fp.stem.replace("_tissue", ""): fp for fp in mask_list}
    # common_keys = case_dict.keys() & mask_dict.keys()
    # sorted_case_list = [case_dict[key] for key in sorted(common_keys)]
    # sorted_mask_list = [mask_dict[key] for key in sorted(common_keys)]
    sorted_case_list = case_list
    sorted_mask_list = mask_list
    return sorted_case_list, sorted_mask_list


def extract_coordinates(wsi_fp, mask_fp, spacing, region_size, num_workers: int = 1):
    wsi = WholeSlideImage(wsi_fp, mask_fp)
    coordinates, tissue_percentages, patch_level, resize_factor = wsi.get_patch_coordinates(spacing, region_size, num_workers=num_workers)
    sorted_coordinates, sorted_tissue_percentages = sort_coords_with_tissue(coordinates, tissue_percentages)
    return sorted_coordinates, sorted_tissue_percentages, patch_level, resize_factor


def save_patch(coord, wsi_fp, spacing, patch_size, resize_factor, backend: str = "asap"):
    region_size = patch_size // resize_factor
    x, y = coord
    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    patch = wsi.get_patch(x, y, patch_size, patch_size, spacing=spacing, center=False)
    pil_patch = Image.fromarray(patch).convert("RGB")
    if resize_factor != 1:
        assert patch_size % region_size == 0, f"width ({patch_size}) is not divisible by region_size ({region_size})"
        pil_patch = pil_patch.resize((region_size, region_size))
    patch_fp = Path(f"/tmp/patches/{wsi_fp.stem}/{int(x)}_{int(y)}.jpg")
    pil_patch.save(patch_fp)
    return patch_fp


def save_patch_mp(args):
    coord, wsi_fp, spacing, patch_size, resize_factor = args
    return save_patch(coord, wsi_fp, spacing, patch_size, resize_factor)


def save_patches(wsi_fp, coord, tissue_pct, patch_level, region_size, factor, backend: str = "asap", nregion_max: Optional[int] = None, num_workers: int = 1):
    wsi_name = wsi_fp.stem
    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    patch_spacing = wsi.spacings[patch_level]
    patch_size = region_size * factor
    patch_dir = Path(f"/tmp/patches/{wsi_name}/")
    patch_dir.mkdir(parents=True, exist_ok=True)
    # filter out patches with low tissue percentage
    coord = keep_top_regions(coord, tissue_pct, nregion_max)
    if nregion_max:
        assert len(coord) <= nregion_max, f"Number of regions ({len(coord)}) is greater than maximum allowed ({nregion_max})"
    if num_workers > 1:
        num_workers = min(mp.cpu_count(), num_workers)
        if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
            num_workers = min(
                num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            )
        iterable = [
            (c, wsi_fp, patch_spacing, patch_size, factor)
            for c in coord
        ]
        with mp.Pool(num_workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(save_patch_mp, iterable),
                desc="Patch saving",
                unit=" patch",
                total=len(iterable),
            ):
                pass
    else:
        with tqdm.tqdm(
            coord,
            desc=f"Saving patches for {wsi_fp.stem}",
            unit=" region",
            leave=False,
        ) as t:
            for c in t:
                save_patch(c, wsi, wsi_name, patch_spacing, patch_size, factor)