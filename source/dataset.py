import torch
import cucim
import cupy as cp
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
        img = transforms.functional.to_tensor(patch)
        return idx, img


class PatchDatasetCucim(torch.utils.data.Dataset):
    def __init__(self, wsi_fp, coord, patch_level, factor, region_size, nfeats_max: Optional[int] = None):
        self.seed = 0
        self.coord = coord
        self.factor = factor
        self.region_size = region_size
        self.wsi = cucim.CuImage(str(wsi_fp))
        self.patch_level = patch_level
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
        patch = self.wsi.read_region((x, y), [self.width, self.height], self.patch_level)
        cupy_array = cp.asarray(patch).astype('float32') / 255.0
        img = torch.as_tensor(cupy_array)
        return idx, img


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