import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                            1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / self.linear.in_features) / self.omega_0,
                                            math.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNet(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=1, depth=3):
        super().__init__()

        layers = []
        layers.append(SineLayer(in_dim, hidden_dim, is_first=True))

        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        return self.final(x)
    
    
def build_siren_input(seg_logits):
    B, C, H, W = seg_logits.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=seg_logits.device),
        torch.linspace(-1, 1, W, device=seg_logits.device),
        indexing='ij'
    )

    coords = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    coords = coords.unsqueeze(0).repeat(B,1,1,1)

    seg = seg_logits.permute(0,2,3,1)  # [B,H,W,C]

    inp = torch.cat([coords, seg], dim=-1)

    return inp.view(B, -1, 2 + C)