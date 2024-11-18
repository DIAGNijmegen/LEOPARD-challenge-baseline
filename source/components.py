import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from einops import rearrange
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from source.vision_transformer import vit_small, vit4k_xs
from source.model_utils import Attn_Net_Gated, update_state_dict
from source.dist_utils import is_main_process


class CustomViT(nn.Module):
    def __init__(
        self,
        pretrained_weights: str,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        embed_dim: int = 384,
        mask_attn: bool = False,
        num_register_tokens: int = 0,
        verbose: bool = True,
    ):
        super(CustomViT, self).__init__()

        self.ps = patch_size

        self.vit = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim,
            mask_attn=mask_attn,
            num_register_tokens=num_register_tokens,
        )

        if Path(pretrained_weights).is_file():
            if verbose and is_main_process():
                print("Loading pretrained weights for ViT-S")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            state_dict, msg = update_state_dict(self.vit.state_dict(), state_dict)
            self.vit.load_state_dict(state_dict, strict=False)
            if verbose and is_main_process():
                print(f"Pretrained weights found at {pretrained_weights}")
                print(msg)

        elif verbose and is_main_process():
            print(
                f"{pretrained_weights} doesnt exist ; please provide path to existing file"
            )

    def forward(self, x, pct: Optional[torch.Tensor] = None, pct_thresh: float = 0.0):
        mask_mini_patch = None
        if pct is not None:
            mask_mini_patch = (pct > pct_thresh).int()  # [num_patches, nminipatch**2]
            # add the [CLS] token to the mask
            cls_token = mask_mini_patch.new_ones((mask_mini_patch.size(dim=0), 1))
            mask_mini_patch = torch.cat(
                (cls_token, mask_mini_patch), dim=1
            )  # [num_patches, num_mini_patches+1]
        # x = [B, 3, region_size, region_size]
        num_patches = (x.shape[2] // self.ps) ** 2
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [B, 3, npatch, region_size, ps] -> [B, 3, npatch, npatch, ps, ps]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [B*num_patches, 3, ps, ps]

        patch_feature = (
            self.vit(x, mask=mask_mini_patch).detach()
        )  # [B*num_patches, 384]
        patch_feature = patch_feature.reshape(-1, num_patches, patch_feature.shape[-1])
        return patch_feature


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_weights: str, region_size: int, patch_size: int = 256):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
        self.load_weights(pretrained_weights)
        self.num_patches = (region_size // patch_size) ** 2
        for param in self.encoder.parameters():
            param.requires_grad = False

    def build_encoder(self):
        raise NotImplementedError

    def load_weights(self, pretrained_weights, verbose: bool = True):
        if Path(pretrained_weights).is_file():
            if verbose and is_main_process():
                print("Loading pretrained weights for UNI")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            state_dict, msg = update_state_dict(self.vit.state_dict(), state_dict)
            self.vit.load_state_dict(state_dict, strict=True)
            if verbose and is_main_process():
                print(f"Pretrained weights found at {pretrained_weights}")
                print(msg)

        elif verbose and is_main_process():
            print(
                f"{pretrained_weights} doesnt exist ; please provide path to existing file"
            )

    def get_transforms(self):
        data_config = resolve_data_config(
            self.encoder.pretrained_cfg, model=self.encoder
        )
        transforms = create_transform(**data_config)
        return transforms

    def forward(self, x):
        patch_feature = self.encoder(x).detach()  # [B*num_patches, out_features_dim]
        patch_feature = patch_feature.reshape(-1, self.num_patches, patch_feature.shape[-1]) # [B, num_patches, out_features_dim]
        return patch_feature


class UNI(FeatureExtractor):
    def __init__(self, pretrained_weights: str, region_size: int, patch_size: int = 256):
        super(UNI, self).__init__(pretrained_weights, region_size, patch_size)
        if patch_size == 256:
            self.encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            self.encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        self.encoder.pretrained_cfg[
            "interpolation"
        ] = "bicubic"

    def build_encoder(self):
        return timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)


class Kaiko(FeatureExtractor):
    def __init__(self, pretrained_weights: str, region_size: int, patch_size: int = 256):
        super(Kaiko, self).__init__(pretrained_weights, region_size, patch_size)

    def build_encoder(self):
        pretrained_cfg = {
            'tag': 'augreg2_in21k_ft_in1k',
            'custom_load': False,
            'input_size': [3, 224, 224],
            'fixed_input_size': True,
            'interpolation': 'bicubic',
            'crop_pct': 0.9,
            'crop_mode': 'center',
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 0,
            'pool_size': None,
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
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
            nn.Linear(hidden_embed_dim, output_embed_dim), nn.ReLU(), nn.Dropout(dropout)
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
            *[nn.Linear(output_embed_dim, output_embed_dim), nn.ReLU(), nn.Dropout(dropout)]
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
                mask_patch = torch.cat((cls_token, register_tokens, mask_patch), dim=1) # [M, num_patches+1+self.num_register_tokens_region]
            else:
                mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [M, num_patches+1]
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
            print(f"{self.pretrained_weights} doesn't exist; please provide path to an existing file")

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