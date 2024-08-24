import time
import tqdm
import torch
import pandas as pd
import torch.nn as nn

from torch.cuda.amp import autocast

from source.dataset import PatchDatasetFromDisk
from source.utils import load_inputs


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
        num_workers_data_loading: int = 1,
        backend: str = "openslide",
    ):

        self.spacing = spacing
        self.region_size = region_size
        self.batch_size = batch_size
        self.num_workers_data_loading = num_workers_data_loading
        self.backend = backend
        self.features_dim = features_dim

        self.npatch = int(region_size // patch_size) ** 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = feature_extractor.to(self.device, non_blocking=True)
        self.feature_extractor.eval()

        self.feature_aggregator = feature_aggregator.to(self.device, non_blocking=True)
        self.feature_aggregator.eval()

    def extract_slide_feature(self, wsi_fp):
        dataset = PatchDatasetFromDisk(wsi_fp)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_data_loading, pin_memory=True)
        M = len(dataset)
        slide_feature = torch.zeros(M, self.npatch, self.features_dim).to(self.device, non_blocking=True)
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
                    for i, j in enumerate(idx):
                        slide_feature[j] = features[i]
                    torch.cuda.empty_cache()
        return slide_feature

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            with autocast():
                feature = self.feature_extractor(patch)
        return feature

    def write_outputs(self, case_list, predictions, nregion, processing_time):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        df = pd.DataFrame({
            "slide_id": [fp.stem for fp in case_list],
            "overall_survival_years": predictions,
            "nregion": nregion,
            "processing_time": processing_time,
        })
        df.to_csv("/output/predictions.csv", index=False)
        return df

    def predict(self, feature):
        with torch.no_grad():
            with autocast():
                logit = self.feature_aggregator(feature)
                hazard = torch.sigmoid(logit)
                surv = torch.cumprod(1 - hazard, dim=1)
                risk = -torch.sum(surv, dim=1).detach().item()
        return risk

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        case_list, _ = load_inputs()
        predictions, nregions, processing_times = [], [], []
        with tqdm.tqdm(
            case_list,
            desc="Inference",
            unit=" case",
            total=len(case_list),
            leave=True,
        ) as t:
            for wsi_fp in t:
                start_time = time.time()
                tqdm.tqdm.write(f"Processing {wsi_fp.stem}")
                feature = self.extract_slide_feature(wsi_fp)
                nregion = len(feature)
                risk = self.predict(feature)
                overall_survival = self.postprocess(risk)
                processing_time = round(time.time() - start_time, 2)
                nregions.append(nregion)
                processing_times.append(processing_time)
                predictions.append(overall_survival)
        self.write_outputs(case_list, predictions, nregions, processing_times)
        return predictions

    def postprocess(self, risk):
        # risk_shifted = risk + abs(min(risk))
        # overall_survival_years = risk_shifted * self.risk_scaling_factor
        overall_survival_years = risk
        return overall_survival_years