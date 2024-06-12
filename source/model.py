import tqdm
import time
import torch
import pandas as pd

from pathlib import Path

from source.utils import sort_coords
from source.wsi import WholeSlideImage
from source.dataset import PatchDataset
from source.components import FeatureExtractor, FeatureAggregator


class HierarchicalViT():
    def __init__(
        self,
        feature_extractor_weights: str,
        feature_aggregator_weights: str,
        spacing: float,
        region_size: int,
        nbins: int = 4,
        batch_size: int = 1,
        num_workers: int = 1,
        backend="openslide",
    ):

        self.spacing = spacing
        self.region_size = region_size
        self.nbins = nbins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.backend = backend

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = FeatureExtractor(
            feature_extractor_weights,
            patch_size=256,
            mini_patch_size=16,
            mask_attn=False,
        ).to(self.device, non_blocking=True)
        self.feature_extractor.eval()

        self.feature_aggregator = FeatureAggregator(
            feature_aggregator_weights,
            num_classes=nbins,
            region_size=region_size,
            patch_size=256,
            mask_attn=False,
        ).to(self.device, non_blocking=True)
        self.feature_aggregator.eval()

    def load_inputs(self):
        """
        Read from /input/
        """
        case_list = [fp for fp in Path("/input/wsi").glob("*.tif")]
        mask_list = [fp for fp in Path("/input/mask").glob("*.tif")]
        coord_list = []
        patch_level_list = []
        resize_factor_list = []
        with tqdm.tqdm(
            zip(case_list, mask_list),
            desc="Extracting region coordinates",
            unit=" case",
            total=len(case_list),
            leave=True,
        ) as t:
            for wsi_fp, mask_fp in t:
                tqdm.tqdm.write(f"Processing {wsi_fp.stem}")
                wsi = WholeSlideImage(wsi_fp, mask_fp)
                coordinates, patch_level, resize_factor = wsi.get_patch_coordinates(self.spacing, self.region_size)
                sorted_coordinates = sort_coords(coordinates)
                coord_list.append(sorted_coordinates)
                patch_level_list.append(patch_level)
                resize_factor_list.append(resize_factor)
        return case_list, coord_list, patch_level_list, resize_factor_list

    def extract_slide_feature(self, wsi_fp, coord, patch_level, factor):
        dataset = PatchDataset(wsi_fp, coord, patch_level, factor, self.region_size, self.backend)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        patch_features = []
        with tqdm.tqdm(
            dataloader,
            desc=f"Processing {wsi_fp.stem}",
            unit=" region",
            unit_scale=self.batch_size,
            leave=False,
        ) as t:
            for imgs in t:
                imgs = imgs.to(self.device, non_blocking=True)
                batch_features = self.extract_patch_feature(imgs)
                patch_features.append(batch_features)
        slide_feature = torch.cat(patch_features, dim=0)
        return slide_feature

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            feature = self.feature_extractor(patch)
            feature = feature.unsqueeze(0)
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
        logit = self.feature_aggregator(feature)
        hazard = torch.sigmoid(logit)
        surv = torch.cumprod(1 - hazard, dim=1)
        risk = -torch.sum(surv, dim=1).detach().item()
        return risk

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        case_list, coord_list, patch_level_list, resize_factor_list = self.load_inputs()
        time.sleep(1)
        predictions = []
        with tqdm.tqdm(
            zip(case_list, coord_list, patch_level_list, resize_factor_list),
            desc="Extracting case-level features",
            unit=" case",
            total=len(case_list),
            leave=True,
        ) as t:
            for wsi_fp, coord, level, factor in t:
                feature = self.extract_slide_feature(wsi_fp, coord, level, factor)
                risk = self.predict(feature)
                overall_survival = self.postprocess(risk)
                predictions.append(overall_survival)
        prediction_df = self.write_outputs(case_list, predictions)
        return prediction_df

    def postprocess(self, risk):
        overall_survival_years = risk
        return overall_survival_years