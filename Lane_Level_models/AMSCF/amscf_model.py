# -*- coding: utf-8 -*-
# amscf_model.py — AMSCF 主干：GCN + (2层)GRU + Soft-Attention + A/相关图可学习融合
import torch
import torch.nn as nn
from layers import TwoLayerGCN, TransposeAttentionFusion, ImprovedAdjacency, sym_norm


class SoftAttentionPooling(nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1, bias=False)

    def forward(self, h_seq):           # (B*N, T, H)
        score = self.scorer(h_seq)      # (B*N, T, 1)
        att = torch.softmax(score, dim=1)
        ctx = (att * h_seq).sum(dim=1)  # (B*N, H)
        return ctx


class AMSCF(nn.Module):
    """
    输入: x ∈ (B, Tin, N, 1)
    输出: Δy ∈ (B, H, N)   —— 注意：模型输出“增量”（Residual），外部再加上 x_last 得到最终预测
    """
    def __init__(self, num_nodes: int, tin: int, horizon: int,
                 gcn_in: int = 1, gcn_h: int = 96, gcn_out: int = 96,
                 gru_hid: int = 192, att_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.N = num_nodes
        self.Tin = tin
        self.H = horizon

        self.ia = ImprovedAdjacency(num_nodes)
        self.beta1 = nn.Parameter(torch.tensor(0.5))  # 融合 FSP
        self.beta2 = nn.Parameter(torch.tensor(0.5))  # 融合 DSP
        self.fusion = TransposeAttentionFusion(num_nodes, d_att=att_dim)
        self.gcn = TwoLayerGCN(gcn_in, gcn_h, gcn_out, use_ln=True)

        # 2层 GRU
        self.gru = nn.GRU(input_size=gcn_out, hidden_size=gru_hid,
                          num_layers=2, dropout=dropout, batch_first=True)
        self.att = SoftAttentionPooling(gru_hid)
        self.fc1 = nn.Linear(gru_hid, gru_hid)
        self.fc2 = nn.Linear(gru_hid, horizon)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def fuse_with_A(self, A: torch.Tensor, C: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(beta)
        return w * A + (1.0 - w) * C

    def forward(self, x: torch.Tensor, A: torch.Tensor, FSP: torch.Tensor, DSP: torch.Tensor):
        # x: (B, Tin, N, 1)
        IA = self.ia(A)                             # (N, N)
        # A 与两源相关图的可学习融合
        C1 = self.fuse_with_A(A, FSP, self.beta1)  # (N, N)
        C2 = self.fuse_with_A(A, DSP, self.beta2)  # (N, N)
        MG = self.fusion(IA, C1, C2)               # (N, N)
        S_hat = sym_norm(MG)                        # (N, N)

        B, T, N, _ = x.shape
        assert N == self.N

        # 逐时刻做 GCN（不要 squeeze 最后一维）
        gcn_feats = []
        for t in range(T):
            xt = x[:, t, :, :]              # (B, N, 1)
            gt = self.gcn(xt, S_hat)        # (B, N, Fg)
            gcn_feats.append(gt)
        H_seq = torch.stack(gcn_feats, dim=1)      # (B, Tin, N, Fg)

        # GRU over time per node
        BN = B * N
        H_seq = H_seq.view(BN, T, -1)             # (B*N, Tin, Fg)
        out, _ = self.gru(H_seq)                  # (B*N, Tin, H)
        ctx = self.att(out)                       # (B*N, H)

        h = self.act(self.fc1(ctx))
        h = self.drop(h)
        dlt = self.fc2(h)                         # (B*N, horizon) —— 预测“增量”
        dlt = dlt.view(B, N, self.H).transpose(1, 2)  # (B, H, N)
        return dlt
