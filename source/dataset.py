import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, patch_size, coordinates_dir, backend, transforms=None):
        self.seed = 0
        self.name = wsi_fp.stem.replace(" ", "_")
        self.wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
        self.coord = self.load_coordinates(coordinates_dir)
        self.patch_size = patch_size
        self.transforms = transforms

    def load_coordinates(self, coordinates_dir):
        return np.load(Path(coordinates_dir, f"{self.name}.npy"))

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        x, y, patch_size_resized, patch_level, resize_factor = self.coord[idx]
        patch_spacing = self.wsi.spacings[patch_level]
        patch = self.wsi.get_patch(x, y, patch_size_resized, patch_size_resized, spacing=patch_spacing, center=False)
        pil_patch = Image.fromarray(patch).convert("RGB")
        if resize_factor != 1:
            assert patch_size_resized % self.patch_size == 0, f"width ({patch_size_resized}) is not divisible by region_size ({self.patch_size})"
            pil_patch = pil_patch.resize((self.patch_size, self.patch_size))
        if self.transforms is not None:
            img = self.transforms(pil_patch)
        else:
            img = pil_patch
        return idx, img


class PatchDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, patch_dir, transforms=None):
        self.seed = 0
        self.name = wsi_fp.stem.replace(" ", "_")
        self.patches = sorted([x for x in Path(patch_dir, self.name).glob("*.jpg")])
        self.transforms = transforms

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        pil_patch = Image.open(self.patches[idx])
        if self.transforms is not None:
            img = self.transforms(pil_patch)
        else:
            img = pil_patch
        return idx, img