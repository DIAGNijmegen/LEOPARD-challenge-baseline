import time
import tqdm
import torch
import pandas as pd
import torch.nn as nn
import torch.distributed as dist

from pathlib import Path
from contextlib import nullcontext

from source.utils import load_inputs
from source.dist_utils import is_main_process


class MIL():
    def __init__(
        self,
        features_dir: Path,
        feature_aggregator: nn.Module,
        mixed_precision: bool = False,
        distributed: bool = False,
    ):

        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.device_id = 0
        if self.distributed:
            self.device = torch.device(f"cuda:{dist.get_rank()}")
            self.device_id = dist.get_rank()
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.features_dir = features_dir

        self.feature_aggregator = feature_aggregator.to(self.device, non_blocking=True)
        self.feature_aggregator.eval()

    def load_feature(self, fp):
        feature = torch.load(fp, map_location=self.device)
        return feature

    def write_outputs(self, features_list, predictions, nregion, processing_time):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        df = pd.DataFrame({
            "case_id": [fp.stem for fp in features_list],
            "overall_survival_years": predictions,
            "nregion": nregion,
            "processing_time": processing_time,
        })
        df.to_csv("/output/predictions.csv", index=False)
        return df

    def predict(self, feature):
        autocast_context = nullcontext()
        if self.mixed_precision:
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        with torch.no_grad():
            with autocast_context:
                logit = self.feature_aggregator(feature)
                hazard = torch.sigmoid(logit)
                surv = torch.cumprod(1 - hazard, dim=1)
                risk = -torch.sum(surv, dim=1).detach().item()
        return risk

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        features_list = load_inputs()
        predictions, nregions, processing_times = [], [], []
        with tqdm.tqdm(
            features_list,
            desc="Inference",
            unit=" case",
            total=len(features_list),
            leave=True,
            disable=not is_main_process(),
            position=0,
        ) as t:
            for fp in t:
                start_time = time.time()
                feature = self.load_feature(fp)
                if self.distributed and feature is None:
                    continue  # skip further processing if this is not the main process
                nregion = len(feature)
                risk = self.predict(feature)
                overall_survival = self.postprocess(risk)
                processing_time = round(time.time() - start_time, 2)
                nregions.append(nregion)
                processing_times.append(processing_time)
                predictions.append(overall_survival)
        if not self.distributed or is_main_process():
            self.write_outputs(features_list, predictions, nregions, processing_times)
        return predictions

    def postprocess(self, risk):
        overall_survival_years = -risk
        return overall_survival_years