import tqdm
import torch
import pandas as pd
import torch.nn as nn

from pathlib import Path
from typing import Optional

from source.utils import sort_coords
from source.wsi import WholeSlideImage
from source.dataset import PatchDataset


class MIL():
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_aggregator: nn.Module,
        spacing: float,
        region_size: int,
        features_dim: int,
        patch_size: int = 256,
        batch_size: int = 1,
        num_workers: int = 1,
        backend: str = "openslide",
        nfeats_max: Optional[int] = None,
    ):

        self.spacing = spacing
        self.region_size = region_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.backend = backend
        self.features_dim = features_dim
        self.nfeats_max = nfeats_max

        self.npatch = int(region_size // patch_size) ** 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = feature_extractor.to(self.device, non_blocking=True)
        self.feature_extractor.eval()

        self.feature_aggregator = feature_aggregator.to(self.device, non_blocking=True)
        self.feature_aggregator.eval()

    def load_inputs(self):
        """
        Read from /input/
        """
        case_list = sorted([fp for fp in Path("/input/images/prostatectomy-wsi").glob("*.tif")])
        mask_list = sorted([fp for fp in Path("/input/images/prostatectomy-tissue-mask").glob("*.tif")])
        return case_list, mask_list

    def extract_coordinates(self, wsi_fp, mask_fp):
        wsi = WholeSlideImage(wsi_fp, mask_fp)
        coordinates, patch_level, resize_factor = wsi.get_patch_coordinates(self.spacing, self.region_size)
        sorted_coordinates = sort_coords(coordinates)
        return sorted_coordinates, patch_level, resize_factor

    def extract_slide_feature(self, wsi_fp, coord, patch_level, factor, nfeats_max: Optional[int] = None):
        dataset = PatchDataset(wsi_fp, coord, patch_level, factor, self.region_size, self.backend, nfeats_max=nfeats_max)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        M = len(dataset)
        slide_feature = torch.zeros(M, self.npatch, self.features_dim)
        with torch.no_grad():
            with tqdm.tqdm(
                dataloader,
                desc=f"Processing {wsi_fp.stem}",
                unit=" region",
                unit_scale=self.batch_size,
                leave=False,
            ) as t:
                for batch in t:
                    idx, img = batch
                    img = img.to(self.device, non_blocking=True)
                    features = self.extract_patch_feature(img)
                    features = features.cpu()
                    for i, j in enumerate(idx):
                        slide_feature[j] = features[i]
                    torch.cuda.empty_cache()
        return slide_feature

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            feature = self.feature_extractor(patch)
        return feature

    def write_outputs(self, case_list, predictions):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        df = pd.DataFrame({
            "slide_id": [fp.stem for fp in case_list],
            "overall_survival_years": predictions,
        })
        df.to_csv("/output/predictions.csv", index=False)
        return df

    def predict(self, feature):
        with torch.no_grad():
            logit = self.feature_aggregator(feature)
            hazard = torch.sigmoid(logit)
            surv = torch.cumprod(1 - hazard, dim=1)
            risk = -torch.sum(surv, dim=1).detach().item()
        return risk

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        case_list, mask_list = self.load_inputs()
        predictions = []
        with tqdm.tqdm(
            zip(case_list, mask_list),
            desc="Inference",
            unit=" case",
            total=len(case_list),
            leave=True,
        ) as t:
            for wsi_fp, mask_fp in t:
                tqdm.tqdm.write(f"Processing {wsi_fp.stem}")
                coord, level, factor = self.extract_coordinates(wsi_fp, mask_fp)
                feature = self.extract_slide_feature(wsi_fp, coord, level, factor, nfeats_max=self.nfeats_max)
                feature = feature.to(self.device, non_blocking=True)
                risk = self.predict(feature)
                overall_survival = self.postprocess(risk)
                predictions.append(overall_survival)
        self.write_outputs(case_list, predictions)
        return predictions

    def postprocess(self, risk):
        overall_survival_years = risk
        return overall_survival_years