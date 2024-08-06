import torch
import wholeslidedata as wsd

from PIL import Image
from torchvision import transforms
from typing import Optional
from pathlib import Path


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, coord, patch_level, factor, region_size, backend, nfeats_max: Optional[int] = None):
        self.seed = 0
        self.coord = coord
        self.factor = factor
        self.region_size = region_size
        self.wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
        self.patch_spacing = self.wsi.spacings[patch_level]
        self.width = self.region_size * factor
        self.height = self.region_size * factor

        if nfeats_max and len(self.coord) > nfeats_max:
            torch.manual_seed(self.seed)
            sampled_indices = torch.randperm(len(self.coord))[:nfeats_max].sort().values
            self.coord = self.coord[sampled_indices]

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        x, y = self.coord[idx]
        patch = self.wsi.get_patch(x, y, self.width, self.height, spacing=self.patch_spacing, center=False)
        pil_patch = Image.fromarray(patch).convert("RGB")
        if self.factor != 1:
            assert self.width % self.region_size == 0, f"width ({self.width}) is not divisible by region_size ({self.region_size})"
            pil_patch = pil_patch.resize((self.region_size, self.region_size))
        img = transforms.functional.to_tensor(pil_patch)
        return img


class PatchDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, nfeats_max: Optional[int] = None):
        self.seed = 0
        self.name = wsi_fp.stem
        self.patches = sorted([x for x in Path(f"/output/patches/{self.name}").glob("*.jpg")])
        if nfeats_max and len(self.patches) > nfeats_max:
            torch.manual_seed(self.seed)
            sampled_indices = torch.randperm(len(self.patches))[:nfeats_max].sort().values
            self.patches = self.patches[sampled_indices]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = Image.open(self.patches[idx])
        img = transforms.functional.to_tensor(patch)
        return img