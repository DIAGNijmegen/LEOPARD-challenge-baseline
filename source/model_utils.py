import torch.nn as nn


def update_state_dict(model_dict, state_dict):
    success, shape_mismatch, missing_keys = 0, 0, 0
    updated_state_dict = {}
    shape_mismatch_list = []
    missing_keys_list = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                updated_state_dict[k] = v
                success += 1
            else:
                updated_state_dict[k] = model_dict[k]
                shape_mismatch += 1
                shape_mismatch_list.append(k)
        else:
            missing_keys += 1
            missing_keys_list.append(k)
    if shape_mismatch > 0 or missing_keys > 0:
        msg = (f"{success}/{len(state_dict)} weight(s) loaded successfully\n"
           f"{shape_mismatch} weight(s) not loaded due to mismatching shapes: {shape_mismatch_list}\n"
           f"{missing_keys} key(s) not found in model: {missing_keys_list}")
    else:
        msg = f"{success}/{len(state_dict)} weight(s) loaded successfully."
    return updated_state_dict, msg


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, num_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            num_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x num_classes
        return A, x
