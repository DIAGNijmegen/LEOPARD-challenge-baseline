import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional

from source.vision_transformer import vit4k_xs
from source.model_utils import Attn_Net_Gated, update_state_dict
from source.dist_utils import is_main_process


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