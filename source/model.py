import time
import tqdm
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.distributed as dist

from contextlib import nullcontext

from source.dataset import PatchDataset, PatchDatasetFromDisk
from source.augmentations import RegionUnfolding
from source.utils import load_inputs
from source.dist_utils import is_main_process


class MIL():
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_aggregator: nn.Module,
        spacing: float,
        region_size: int,
        features_dim: int,
        coordinates_dir: str,
        patch_dir: str,
        patch_size: int = 256,
        batch_size: int = 1,
        mixed_precision: bool = False,
        load_patches_from_disk: bool = False,
        num_workers_data_loading: int = 1,
        backend: str = "openslide",
        distributed: bool = False,
    ):

        self.spacing = spacing
        self.region_size = region_size
        self.batch_size = batch_size
        self.num_workers_data_loading = num_workers_data_loading
        self.backend = backend
        self.features_dim = features_dim
        self.coordinates_dir = coordinates_dir
        self.load_patches_from_disk = load_patches_from_disk
        if self.load_patches_from_disk:
            assert patch_dir is not None, "patch_dir must be provided when load_patches_from_disk is True"
        self.patch_dir = patch_dir

        self.npatch = int(region_size // patch_size) ** 2

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

        self.autocast_context = nullcontext()
        if mixed_precision:
            self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

        self.feature_extractor = feature_extractor.to(self.device, non_blocking=True)
        self.feature_extractor.eval()

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                RegionUnfolding(patch_size),
                self.feature_extractor.get_transforms(),
            ]
        )

        self.feature_aggregator = feature_aggregator.to(self.device, non_blocking=True)
        self.feature_aggregator.eval()

    def extract_slide_feature(self, wsi_fp):
        if self.load_patches_from_disk:
            dataset = PatchDatasetFromDisk(wsi_fp, self.patch_dir, self.transforms)
        else:
            dataset = PatchDataset(
                wsi_fp,
                coordinates_dir=self.coordinates_dir,
                backend=self.backend,
                transforms=self.transforms,
            )
        if self.distributed:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers_data_loading, pin_memory=True)
        patch_feature = torch.empty((0, self.npatch, self.features_dim), device=self.device)
        patch_indices = torch.empty((0,), dtype=torch.long, device=self.device)
        with torch.no_grad():
            with tqdm.tqdm(
                dataloader,
                desc=f"GPU {self.device_id}: {wsi_fp.stem}",
                unit=" region",
                unit_scale=self.batch_size,
                leave=False,
                position=1+self.device_id,
            ) as t:
                for batch in t:
                    idx, img = batch
                    img = img.to(self.device, non_blocking=True)
                    features = self.extract_patch_feature(img)
                    patch_feature = torch.cat((patch_feature, features), dim=0)
                    patch_indices = torch.cat((patch_indices, idx.to(self.device, non_blocking=True)), dim=0)
                    torch.cuda.empty_cache()

        if self.distributed:
            dist.barrier()

        if self.distributed:
            # gather features and indices from all GPUs
            gathered_feature = [torch.zeros_like(patch_feature, device=self.device) for _ in range(dist.get_world_size())]
            gathered_indices = [torch.zeros_like(patch_indices, device=self.device) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_feature, patch_feature)
            dist.all_gather(gathered_indices, patch_indices)
            if is_main_process():
                # concatenate the gathered features and indices
                slide_feature = torch.cat(gathered_feature, dim=0)
                patch_indices = torch.cat(gathered_indices, dim=0)
                # remove duplicates
                unique_indices = torch.unique(patch_indices)
                # create a final tensor to store the features in the correct order
                slide_feature_ordered = torch.zeros((len(unique_indices), self.npatch, self.features_dim), device=self.device)
                # insert each feature into its correct position based on patch_indices
                slide_feature_ordered[unique_indices] = slide_feature[unique_indices]
            else:
                slide_feature_ordered = None
        else:
            slide_feature_ordered = torch.zeros_like(patch_feature, device=self.device)
            slide_feature_ordered[patch_indices] = patch_feature

        return slide_feature_ordered

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            with self.autocast_context:
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
            with self.autocast_context:
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
            disable=not is_main_process(),
            position=0,
        ) as t:
            for wsi_fp in t:
                start_time = time.time()
                feature = self.extract_slide_feature(wsi_fp)
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
            self.write_outputs(case_list, predictions, nregions, processing_times)
        return predictions

    def postprocess(self, risk):
        # risk_shifted = risk + abs(min(risk))
        # overall_survival_years = risk_shifted * self.risk_scaling_factor
        overall_survival_years = -risk
        return overall_survival_years