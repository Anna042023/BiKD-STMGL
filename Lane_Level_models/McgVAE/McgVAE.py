import torch
import torch.nn as nn
import torch.nn.functional as F
from masking import mask_lane_noise_speeds
from Imputer import LaneSpeedImputer
from Graph_Attention import DynamicGraphConvNet

# ---------------- RevIN（按时间维做标准化，保留每节点统计） ----------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self._cached = False

    def forward(self, x, mode: str):
        # x: (B, T, N)
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)  # (B,1,N)
            self.std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)  # (B,1,N)
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
            self._cached = True
            return x
        elif mode == "denorm":
            if not self._cached:
                raise RuntimeError("RevIN: call norm() before denorm().")
            if self.affine:
                x = (x - self.bias.view(1, 1, -1)) / (self.weight.view(1, 1, -1) + self.eps)
            x = x * self.std + self.mean
            return x
        else:
            raise NotImplementedError


# ---------------- Encoder：取末尾 horizon 段，喂给动态图注意力 ----------------
class Encoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        # Graph_Attention.DynamicGraphConvNet 约定输入形状 (B, H, N)
        self.graph = DynamicGraphConvNet(node_features=1, horizon=horizon, num_nodes=num_nodes)
        self.fc_mu   = nn.Linear(horizon * num_nodes, hidden_dim)
        self.fc_logv = nn.Linear(horizon * num_nodes, hidden_dim)

    def forward(self, x_bn):  # x_bn: (B, T, N)
        x_last = x_bn[:, -self.horizon:, :]       # (B, H, N)
        x_g = self.graph(x_last)                  # (B, N, H)
        x_flat = x_g.transpose(1, 2).reshape(x_g.size(0), -1)  # (B, H*N)
        mu   = self.fc_mu(x_flat)
        logv = self.fc_logv(x_flat)
        return mu, logv


# ---------------- Decoder：hidden -> horizon，复制到每个节点 ----------------
class Decoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, horizon)

    def forward(self, z):   # z: (B, hidden)
        h = F.relu(self.fc1(z))                         # (B, hidden)
        h = h.unsqueeze(1).repeat(1, self.num_nodes, 1) # (B, N, hidden)
        y = self.fc2(h)                                 # (B, N, H)
        return y


# ---------------- McgVAE 主体 ----------------
class McgVAE(nn.Module):
    def __init__(self,
                 seq_len: int,
                 num_road_nodes: int,
                 num_lane_nodes: int,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 horizon: int,
                 road_lane_count=None,
                 corr_threshold: float = 2.0):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_road_nodes = num_road_nodes
        self.num_lane_nodes = num_lane_nodes
        self.road_lane_count = road_lane_count or [5,5,5,5,5,5,5,5]  # 40节点默认
        self.corr_threshold = corr_threshold

        # RevIN
        self.revin_road = RevIN(num_road_nodes)
        self.revin_lane = RevIN(num_lane_nodes)

        # 编解码器
        self.enc_road = Encoder(num_nodes=num_road_nodes, hidden_dim=hidden_dim, horizon=horizon)
        self.enc_lane = Encoder(num_nodes=num_lane_nodes, hidden_dim=hidden_dim, horizon=horizon)
        self.dec_road = Decoder(num_nodes=num_road_nodes, hidden_dim=hidden_dim, horizon=horizon)
        self.dec_lane = Decoder(num_nodes=num_lane_nodes, hidden_dim=hidden_dim, horizon=horizon)

        # 修复器
        self.imputer = LaneSpeedImputer(seq_len, num_lane_nodes, num_road_nodes, hidden_dim)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, road_x, lane_x):
        """
        road_x : (B, T, R, 1)
        lane_x : (B, T, L, 1)
        return : road_pred (B, R, H), lane_pred (B, L, H)
        """
        B, T, R, _ = road_x.shape
        _, _, L, _ = lane_x.shape
        road_x = road_x.view(B, T, R)
        lane_x = lane_x.view(B, T, L)

        # 掩码（顺序不一致会报错，捕获后跳过）
        try:
            lane_x = mask_lane_noise_speeds(lane_x, road_x, self.road_lane_count, self.corr_threshold)
        except Exception as e:
            print(f"[WARNING] mask_lane_noise_speeds failed: {e}\nSkipping masking.")
        lane_x = self.imputer(lane_x, road_x, self.road_lane_count)  # (B,T,L)

        # RevIN
        road_bn = self.revin_road(road_x, "norm")  # (B,T,R)
        lane_bn = self.revin_lane(lane_x, "norm")  # (B,T,L)

        # 编码->采样->解码
        mu_r, logv_r = self.enc_road(road_bn)
        mu_l, logv_l = self.enc_lane(lane_bn)
        zr = self.reparam(mu_r, logv_r)   # (B, hidden)
        zl = self.reparam(mu_l, logv_l)   # (B, hidden)

        road_y_bn = self.dec_road(zr)     # (B,R,H)
        lane_y_bn = self.dec_lane(zl)     # (B,L,H)

        # 用输入统计量反归一化：先转 (B,H,N) 做 denorm，再转回 (B,N,H)
        road_den_in = road_y_bn.transpose(1, 2)     # (B,H,R)
        lane_den_in = lane_y_bn.transpose(1, 2)     # (B,H,L)
        road_den = self.revin_road(road_den_in, "denorm")  # (B,H,R)
        lane_den = self.revin_lane(lane_den_in, "denorm")  # (B,H,L)
        road_pred = road_den.transpose(1, 2)        # (B,R,H)
        lane_pred = lane_den.transpose(1, 2)        # (B,L,H)
        return road_pred, lane_pred
