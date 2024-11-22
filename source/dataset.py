import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, coordinates_dir, backend, transforms=None):
        self.path = wsi_fp
        self.name = wsi_fp.stem.replace(" ", "_")
        self.backend = backend
        self.coord = self.load_coordinates(coordinates_dir)
        self.transforms = transforms

    def load_coordinates(self, coordinates_dir):
        return np.load(Path(coordinates_dir, f"{self.name}.npy"))

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        wsi = wsd.WholeSlideImage(self.path, backend=self.backend)
        x, y, patch_size_resized, patch_level, resize_factor = self.coord[idx]
        patch_spacing = wsi.spacings[patch_level]
        patch = wsi.get_patch(x, y, patch_size_resized, patch_size_resized, spacing=patch_spacing, center=False)
        pil_patch = Image.fromarray(patch).convert("RGB")
        if resize_factor != 1:
            patch_size = int(patch_size_resized / resize_factor)
            assert patch_size_resized % patch_size == 0, f"width ({patch_size_resized}) is not divisible by region_size ({patch_size})"
            pil_patch = pil_patch.resize((patch_size, patch_size))
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