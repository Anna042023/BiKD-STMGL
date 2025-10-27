import torch
import torch.nn as nn
from model import gwnet
from util import masked_mape, masked_rmse


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate,
                 wdecay, device, supports, gcn_bool=True, addaptadj=True, aptinit=None):

        self.scaler = scaler
        self.device = device

        self.model = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=seq_length,           # horizon
            residual_channels=nhid
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_fn = nn.L1Loss()

    def _postprocess(self, out):
        pred = out.transpose(3, 1)                 # (B, 1, N, horizon)
        pred = self.scaler.inverse_transform(pred) # 反标准化（保持梯度）
        pred = pred.transpose(3, 1)                # (B, horizon, N, 1)
        return pred

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        # (B, Tin, N, C) -> (B, C, N, Tin)
        x = input.permute(0, 3, 2, 1).contiguous()
        x = nn.functional.pad(x, (1, 0, 0, 0))     # 在时间维左侧 pad 1
        out = self.model(x)                        # (B, horizon, N, 1)
        pred = self._postprocess(out)              # (B, horizon, N, 1)

        loss = self.loss_fn(pred, real_val)        # L1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
        self.optimizer.step()

        mape = masked_mape(pred, real_val).item()
        rmse = masked_rmse(pred, real_val).item()
        return loss.item(), mape, rmse

    @torch.no_grad()
    def eval(self, input, real_val):
        self.model.eval()
        x = input.permute(0, 3, 2, 1).contiguous()
        x = nn.functional.pad(x, (1, 0, 0, 0))
        out = self.model(x)
        pred = out.transpose(3, 1)                 # (B, 1, N, horizon)
        pred = self.scaler.inverse_transform(pred) # 反标准化
        pred = pred.transpose(3, 1)                # (B, horizon, N, 1)

        loss = self.loss_fn(pred, real_val).item()
        mape = masked_mape(pred, real_val).item()
        rmse = masked_rmse(pred, real_val).item()
        return loss, mape, rmse
