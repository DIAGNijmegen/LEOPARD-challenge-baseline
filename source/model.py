import tqdm
import torch
import pandas as pd

from pathlib import Path

from source.utils import sort_coords, track_vram_usage
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
        print("=+=" * 10)

        self.feature_aggregator = FeatureAggregator(
            feature_aggregator_weights,
            num_classes=nbins,
            region_size=region_size,
            patch_size=256,
            mask_attn=False,
        ).to(self.device, non_blocking=True)
        self.feature_aggregator.eval()
        print("=+=" * 10)

    def load_inputs(self):
        """
        Read from /input/
        """
        case_list = sorted([fp for fp in Path("/input/wsi").glob("*.tif")])
        mask_list = sorted([fp for fp in Path("/input/mask").glob("*.tif")])
        case_dict = {fp.stem: fp for fp in case_list}
        mask_dict = {fp.stem.replace("_tissue", ""): fp for fp in mask_list}
        common_keys = case_dict.keys() & mask_dict.keys()
        sorted_case_list = [case_dict[key] for key in sorted(common_keys)]
        sorted_mask_list = [mask_dict[key] for key in sorted(common_keys)]
        return sorted_case_list, sorted_mask_list

    def extract_coordinates(self, wsi_fp, mask_fp):
        wsi = WholeSlideImage(wsi_fp, mask_fp)
        coordinates, patch_level, resize_factor = wsi.get_patch_coordinates(self.spacing, self.region_size)
        sorted_coordinates = sort_coords(coordinates)
        return sorted_coordinates, patch_level, resize_factor

    def extract_slide_feature(self, wsi_fp, coord, patch_level, factor):
        dataset = PatchDataset(wsi_fp, coord, patch_level, factor, self.region_size, self.backend)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        # import wholeslidedata as wsd
        # from PIL import Image
        # from torchvision import transforms
        # wsi = wsd.WholeSlideImage(wsi_fp, backend=self.backend)
        # patch_spacing = wsi.spacings[patch_level]
        # width = self.region_size * factor
        # height = self.region_size * factor
        # dataloader = coord
        patch_features = []
        with torch.no_grad():
            with tqdm.tqdm(
                dataloader,
                desc=f"Processing {wsi_fp.stem}",
                unit=" region",
                unit_scale=self.batch_size,
                leave=False,
            ) as t:
                # for x, y in t:
                #     patch = wsi.get_patch(x, y, width, height, spacing=patch_spacing, center=False)
                #     pil_patch = Image.fromarray(patch).convert("RGB")
                #     if factor != 1:
                #         assert width % self.region_size == 0, f"width ({width}) is not divisible by region_size ({self.region_size})"
                #         pil_patch = pil_patch.resize((self.region_size, self.region_size))
                #     imgs = transforms.functional.to_tensor(pil_patch)
                #     imgs = imgs.unsqueeze(0)
                for imgs in t:
                    imgs = imgs.to(self.device, non_blocking=True)
                    batch_features = self.extract_patch_feature(imgs)
                    batch_features = batch_features.cpu()
                    patch_features.append(batch_features)
                    torch.cuda.empty_cache()
        slide_feature = torch.cat(patch_features, dim=0)
        return slide_feature

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            feature = self.feature_extractor(patch)
            feature = feature.unsqueeze(0)
        return feature

    def write_outputs(self, case_list, predictions, vram_consumption):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        df = pd.DataFrame({
            "slide_id": [fp.stem for fp in case_list],
            "overall_survival_years": predictions,
            "vram": vram_consumption,
        })
        df.to_csv("/output/predictions.csv", index=False)
        return df

    def predict(self, feature):
        with torch.no_grad():
            logit, vram = track_vram_usage(self.feature_aggregator, feature)
            hazard = torch.sigmoid(logit)
            surv = torch.cumprod(1 - hazard, dim=1)
            risk = -torch.sum(surv, dim=1).detach().item()
        return risk, vram

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        case_list, mask_list = self.load_inputs()
        vram_consumption = []
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
                feature = self.extract_slide_feature(wsi_fp, coord, level, factor)
                feature = feature.to(self.device, non_blocking=True)
                risk, vram = self.predict(feature)
                overall_survival = self.postprocess(risk)
                predictions.append(overall_survival)
                vram_consumption.append(vram)
        prediction_df = self.write_outputs(case_list, predictions, vram_consumption)
        return prediction_df

    def postprocess(self, risk):
        overall_survival_years = risk
        return overall_survival_years