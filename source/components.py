import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from torchvision import transforms

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import source.vision_transformer as vits

from source.vision_transformer import vit4k_xs
from source.model_utils import Attn_Net_Gated, update_state_dict
from source.dist_utils import is_main_process
from source.augmentations import make_normalize_transform, MaybeToTensor


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_weights: str, ckpt_key: Optional[str] = None):
        super(FeatureExtractor, self).__init__()
        self.ckpt_key = ckpt_key
        self.encoder = self.build_encoder()
        self.load_weights(pretrained_weights)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def build_encoder(self):
        raise NotImplementedError

    def load_weights(self, pretrained_weights, verbose: bool = True):
        if Path(pretrained_weights).is_file():
            if verbose and is_main_process():
                print(f"Loading encoder weights from: {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if self.ckpt_key:
                state_dict = state_dict[self.ckpt_key]
            nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state_dict, prefix="module."
            )
            nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state_dict, prefix="backbone."
            )
            state_dict, msg = update_state_dict(self.encoder.state_dict(), state_dict)
            if verbose and is_main_process():
                print(msg)
            self.encoder.load_state_dict(state_dict, strict=True)

        elif verbose and is_main_process():
            print(
                f"{pretrained_weights} doesnt exist ; please provide path to existing file"
            )

    def get_transforms(self):
        if self.config:
            data_config = resolve_data_config(self.config)
            transforms = create_transform(**data_config)
        else:
            transforms = None
        return transforms

    def forward(self, x):
        # x = [B, num_patches, 3, 224, 224]
        bs, num_patches = x.shape[0], x.shape[1]
        x = x.reshape(bs * num_patches, *x.shape[2:])  # [B*num_patches, 3, 224, 224]
        patch_feature = self.encoder(x).detach()  # [B*num_patches, out_features_dim]
        patch_feature = patch_feature.reshape(
            bs, num_patches, -1
        )  # [B, num_patches, out_features_dim]
        return patch_feature


class DINOViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 256,
        patch_size: int = 14,
        ckpt_key: str = "teacher",
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        self.input_size = input_size
        self.patch_size = patch_size
        arch2dim = {"vit_large": 1024, "vit_base": 768, "vit_small": 384}
        super(DINOViT, self).__init__(pretrained_weights, ckpt_key)
        self.features_dim = arch2dim[arch]

    def build_encoder(self):
        encoder = vits.__dict__[self.arch](
            img_size=self.input_size, patch_size=self.patch_size
        )
        return encoder

    def get_transforms(self):
        if self.input_size > 224:
            transform = transforms.Compose(
                [
                    MaybeToTensor(),
                    transforms.CenterCrop(224),
                    make_normalize_transform(),
                ]
            )
        else:
            transforms.Compose(
                [
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )
        return transform


class UNI(FeatureExtractor):
    def __init__(self, pretrained_weights: str, patch_size: int = 256):
        self.config = {
            "model_name": "vit_large_patch16_224",
            "patch_size": 16,
            "img_size": 224,
            "init_values": 1.0,
            "num_classes": 0,
            "dynamic_img_size": True,
            "pretrained_cfg": {
                "tag": "uni_mass100k",
                "custom_load": True,
                "crop_pct": 1,
                "input_size": [3, 224, 224],
                "fixed_input_size": False,
                "interpolation": "bilinear",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 0,
                "pool_size": None,
                "first_conv": "patch_embed.proj",
                "classifier": "head",
            },
        }
        if patch_size == 256:
            self.config["pretrained_cfg"]["crop_pct"] = (
                224 / 256
            )  # ensure Resize is 256
        super(UNI, self).__init__(pretrained_weights)
        self.features_dim = 1024

    def build_encoder(self):
        return timm.create_model(**self.config)


class Kaiko(FeatureExtractor):
    def __init__(
        self, pretrained_weights: str, region_size: int, patch_size: int = 256
    ):
        super(Kaiko, self).__init__(pretrained_weights, region_size, patch_size)
        self.features_dim = 768

    def build_encoder(self):
        pretrained_cfg = {
            "tag": "augreg2_in21k_ft_in1k",
            "custom_load": False,
            "input_size": [3, 224, 224],
            "fixed_input_size": True,
            "interpolation": "bicubic",
            "crop_pct": 0.9,
            "crop_mode": "center",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
        }
        return timm.create_model("vit_base_patch16_224", pretrained_cfg=pretrained_cfg)


class HierarchicalViT(nn.Module):
    def __init__(
        self,
        pretrained_weights: str,
        num_classes: int,
        region_size: int,
        patch_size: int = 256,
        input_embed_dim: int = 384,
        hidden_embed_dim: int = 192,
        output_embed_dim: int = 192,
        dropout: float = 0.25,
        mask_attn: bool = False,
        num_register_tokens: int = 0,
    ):
        super(HierarchicalViT, self).__init__()
        self.pretrained_weights = pretrained_weights
        self.npatch = int(region_size // patch_size)
        self.num_register_tokens = num_register_tokens

        self.vit = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=input_embed_dim,
            output_embed_dim=hidden_embed_dim,
            mask_attn=mask_attn,
            img_size_pretrained=region_size,
            num_register_tokens=num_register_tokens,
        )

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(hidden_embed_dim, output_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_embed_dim,
                nhead=3,
                dim_feedforward=output_embed_dim,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=output_embed_dim, D=output_embed_dim, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[
                nn.Linear(output_embed_dim, output_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        self.classifier = nn.Linear(output_embed_dim, num_classes)

        self.load_weights()

    def forward(self, x, pct: Optional[torch.Tensor] = None, pct_thresh: float = 0.0):
        mask_patch = None
        if pct is not None:
            pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
            mask_patch = (pct_patch > pct_thresh).int()  # (M, npatch**2) e.g. (M, 64)
            # add the [CLS] token to the mask
            cls_token = mask_patch.new_ones((mask_patch.size(0), 1))
            # eventually add register tokens to the mask
            # they're added after the [CLS] token in the input sequence
            if self.num_register_tokens_region:
                register_tokens = mask_patch.new_ones(
                    (mask_patch.size(0), self.num_register_tokens_region)
                )
                mask_patch = torch.cat(
                    (cls_token, register_tokens, mask_patch), dim=1
                )  # [M, num_patches+1+self.num_register_tokens_region]
            else:
                mask_patch = torch.cat(
                    (cls_token, mask_patch), dim=1
                )  # [M, num_patches+1]
        # x = [M, 256, 384]
        x = self.vit(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2),
            mask=mask_patch,
        )  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits

    def load_weights(self):
        if self.pretrained_weights and Path(self.pretrained_weights).is_file():
            if is_main_process():
                print("Loading pretrained weights for HViT-XS")
            state_dict = torch.load(self.pretrained_weights, map_location="cpu")
            state_dict, msg = update_state_dict(self.state_dict(), state_dict)
            self.load_state_dict(state_dict, strict=False)
            if is_main_process():
                print(f"Pretrained weights found at {self.pretrained_weights}")
                print(msg)
        elif is_main_process():
            print(
                f"{self.pretrained_weights} doesn't exist; please provide path to an existing file"
            )

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str
