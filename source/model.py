import torch
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path

from source.wsi import WholeSlideImage
from source.components import FeatureExtractor, FeatureAggregator


class HierarchicalViT():
    def __init__(
        self,
        feature_extractor_weights: str,
        feature_aggregator_weights: str,
        spacing: float,
        patch_size: int,
        backend="openslide",
    ):

        self.spacing = spacing
        self.patch_size = patch_size
        self.backend = backend

        self.feature_extractor = FeatureExtractor(
            feature_extractor_weights,
            patch_size=256,
            mini_patch_size=16,
            mask_attn=False,
        )

        nbins = 4
        self.feature_aggregator = FeatureAggregator(
            feature_aggregator_weights,
            num_classes=nbins,
            region_size=4096,
            patch_size=256,
            mask_attn=False,
        )

    def load_inputs(self, spacing: float, patch_size: int):
        """
        Read from /input/
        """
        slide_list = [fp for fp in Path("/input/wsi").glob("*.tif")]
        mask_list = [fp for fp in Path("/input/mask").glob("*.tif")]
        coord_list = []
        patch_level_list = []
        resize_factor_list = []
        for wsi_fp, mask_fp in zip(slide_list, mask_list):
            wsi = WholeSlideImage(wsi_fp, mask_fp)
            coordinates, patch_level, resize_factor = wsi.get_patch_coordinates(wsi, spacing, patch_size)
            coord_list.append(coordinates)
            patch_level_list.append(patch_level)
            resize_factor_list.append(resize_factor)
        return slide_list, coord_list, patch_level_list, resize_factor_list

    def extract_features(self, slide_list, coord_list, patch_level_list, resize_factor_list):
        features = []
        for wsi_fp, coord, level, factor in zip(slide_list, coord_list, patch_level_list, resize_factor_list):
            feature = self.extract_slide_feature(wsi_fp, coord, level, factor)
            features.append(feature)
        return features

    def extract_slide_feature(self, wsi_fp, coord, patch_level, factor):
        patch_features = []
        for x, y in coord:
            wsi = wsd.WholeSlideImage(wsi_fp, backend=self.backend)
            patch_spacing = wsi.spacings[patch_level]
            width, height = int(self.patch_size / factor), int(self.patch_size / factor)
            patch = wsi.get_patch(x, y, width, height, spacing=patch_spacing, center=False)
            pil_patch = Image.fromarray(patch).convert("RGB")
            if factor != 1:
                assert width % self.patch_size == 0
                pil_patch = pil_patch.resize((self.patch_size, self.patch_size))
            patch_feature = self.extract_patch_feature(patch)
            patch_features.append(patch_feature)
        slide_feature = torch.cat(patch_features, dim=0)
        return slide_feature

    def extract_patch_feature(self, patch):
        with torch.no_grad():
            feature = self.feature_extractor(patch)
        return feature

    def write_outputs(self, outputs):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        pass

    def predict(self, features):
        outputs = []
        for feature in features:
            output = self.feature_aggregator(feature)
            outputs.append(output)
        return outputs

    def process(self, spacing: float, region_size: int):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        slide_list, coord_list, patch_level_list, resize_factor_list = self.load_inputs(spacing, region_size)
        features = self.extract_features(slide_list, coord_list, patch_level_list, resize_factor_list)
        outputs = self.predict(features)
        self.write_outputs(slide_list, outputs)

    def postprocess(self, risk):
        overall_survival_years = risk
        return overall_survival_years