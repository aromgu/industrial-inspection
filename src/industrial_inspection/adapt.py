"""Learned per-layer token projection on top of frozen backbone features (scripts/wp_adapt.py trains it)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Proj(nn.Module):
    """Residual MLP on a token: x + W2 gelu(W1 LN(x)); W2 starts at zero, so the projection is the identity at init."""

    def __init__(self, c: int, h: int):
        super().__init__()
        self.ln, self.w1, self.w2 = nn.LayerNorm(c), nn.Linear(c, h), nn.Linear(h, c)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(self.ln(x))))


def load_proj(path: str, c: int = 1280, device: str = "cuda") -> nn.ModuleList:
    ck = torch.load(path, map_location=device)
    m = nn.ModuleList([Proj(c, ck["hidden"]) for _ in ck["layers"]]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval().requires_grad_(False)
