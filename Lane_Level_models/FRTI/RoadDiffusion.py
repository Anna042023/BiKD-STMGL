import torch
import torch.nn as nn

# --------- Chebyshev 图卷积 ---------
class ChebGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, supports):
        # x: (B, N, C)
        B, N, C_in = x.shape
        out = 0
        for k in range(self.K):
            S = supports[k]    # (N, N)
            Sx = torch.matmul(S, x)  # (B, N, C_in)
            out += torch.einsum('bnc,co->bno', Sx, self.theta[k])
        return out


# --------- Temporal 1D Conv Block ---------
class TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B*N, C, T)
        y = self.conv(x)
        y = y[..., :x.size(2)]  # 保持长度一致
        return self.dropout(self.act(y))


# --------- ST Block（修正版：逻辑严密、无越界 reshape）---------
class STBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, K, dropout=0.0):
        super().__init__()
        self.temp1 = TemporalConv(in_ch, hidden_ch, kernel_size=3, dropout=dropout)
        self.gc = ChebGraphConv(hidden_ch, hidden_ch, K)
        self.temp2 = TemporalConv(hidden_ch, out_ch, kernel_size=3, dropout=dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x, supports):
        # x: (B, N, T, C)
        B, N, T, C = x.shape

        # 第一次时间卷积
        h = x.permute(0, 1, 3, 2).reshape(B * N, C, T)
        h = self.temp1(h)
        T_new = h.shape[2]
        h = h.reshape(B, N, -1, T_new).permute(0, 1, 3, 2)  # (B, N, T_new, hidden_ch)

        # 图卷积（对每个时间步单独做）
        out_list = []
        for t in range(T_new):
            xt = h[:, :, t, :]  # (B, N, hidden_ch)
            out_list.append(self.gc(xt, supports))
        h = torch.stack(out_list, dim=2)  # (B, N, T_new, hidden_ch)

        # 第二次时间卷积
        B, N, T2, C2 = h.shape
        h = h.permute(0, 1, 3, 2).reshape(B * N, C2, T2)
        h = self.temp2(h)
        T3 = h.shape[2]
        h = h.reshape(B, N, -1, T3).permute(0, 1, 3, 2)

        # 残差连接
        res = x.permute(0, 1, 3, 2).reshape(B * N, C, T)
        res = self.residual(res)[..., -T3:]
        res = res.reshape(B, N, -1, T3).permute(0, 1, 3, 2)

        return self.act(h + res)


# --------- 轻量 STGCN ---------
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, hidden_dim, supports, dropout=0.1):
        super().__init__()
        self.N = num_nodes
        self.Tin = in_steps
        self.Tout = out_steps
        self.supports = supports

        self.block1 = STBlock(1, hidden_dim, hidden_dim, K=len(supports), dropout=dropout)
        self.block2 = STBlock(hidden_dim, hidden_dim, hidden_dim, K=len(supports), dropout=dropout)

        # (B, C, N, T) -> (B, H, N, T)；随后我们会“取最后一个时间步”
        self.head = nn.Conv2d(hidden_dim, out_steps, kernel_size=(1, 1))

    def forward(self, x):
        # x: (B, T, N)
        B, T, N = x.shape
        assert N == self.N
        x = x.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (B, N, T, 1)

        h = self.block1(x, self.supports)   # (B, N, T1, C)
        h = self.block2(h, self.supports)   # (B, N, T2, C)

        # 变换到 (B, C, N, T2)
        h = h.permute(0, 3, 1, 2).contiguous()   # (B, C, N, T2)

        # 经过 1x1 head 卷积得到 (B, H, N, T2)
        y = self.head(h)                         # (B, H, N, T2)

        # ✅ 关键修复：取“最后一个时间步”的输出，变成三维 (B, H, N)
        y = y[:, :, :, -1]                       # (B, H, N)

        # 再转成 (B, N, H) 与主程序对齐
        return y.permute(0, 2, 1).contiguous()   # (B, N, H)

