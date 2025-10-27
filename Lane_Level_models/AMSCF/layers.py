# -*- coding: utf-8 -*-
# layers.py — 图卷积、注意力融合、改进邻接等组件
import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_norm(adj: torch.Tensor) -> torch.Tensor:
    # Â = D^{-1/2} A D^{-1/2}
    eps = 1e-12
    deg = adj.sum(-1) + eps
    d_inv_sqrt = torch.pow(deg, -0.5)
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(0)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        # x: (B, N, Fin) ; S: (N, N)
        x = self.lin(x)             # -> (B, N, Fout)
        return torch.matmul(S, x)   # 广播: (N,N) @ (B,N,Fout) -> (B,N,Fout)


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, use_ln: bool = True):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hid)
        self.g2 = GCNLayer(hid, out_dim)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, S_hat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.g1(x, S_hat))
        out = self.g2(h, S_hat)
        if self.use_ln:
            out = self.ln(out)
        return out


class TransposeAttentionFusion(nn.Module):
    """
    转置注意力映射融合  C1 与 C2（已与 A 做 β-融合的两源相关图）：
    T = softmax((IA Wq)(IA Wk)^T / sqrt(d))
    MG = σ(α) * (T @ C1) + (1-σ(α)) * (T @ C2)
    """
    def __init__(self, num_nodes: int, d_att: int = 32):
        super().__init__()
        self.Wq = nn.Linear(num_nodes, d_att, bias=False)
        self.Wk = nn.Linear(num_nodes, d_att, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, IA: torch.Tensor, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(IA)                           # (N, d)
        K = self.Wk(IA)                           # (N, d)
        att = torch.softmax(Q @ K.t() / (Q.size(-1) ** 0.5), dim=-1)  # (N, N)
        T1 = att @ C1
        T2 = att @ C2
        alpha = torch.sigmoid(self.alpha)
        MG = alpha * T1 + (1 - alpha) * T2
        MG = torch.relu(MG)
        MG = MG + torch.eye(MG.size(0), device=MG.device)
        return MG


class ImprovedAdjacency(nn.Module):
    """
    改进邻接 IA：在 A 基础上加入可学习的“横向近邻”强化（a,b）
    没有明确车道分组标注时，用编号相邻近似。
    """
    def __init__(self, num_nodes: int):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.2))  # 出
        self.b = nn.Parameter(torch.tensor(0.2))  # 入
        self.N = num_nodes

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        N = self.N
        A = torch.relu(A)
        idx = torch.arange(N, device=A.device)
        left = torch.clamp(idx - 1, 0, N - 1)
        right = torch.clamp(idx + 1, 0, N - 1)
        L = torch.zeros_like(A)
        L[idx, left] = 1.0
        L[idx, right] = 1.0
        IA = A + torch.sigmoid(self.a) * L + torch.sigmoid(self.b) * L.t()
        IA = IA + torch.eye(N, device=A.device)
        return IA
