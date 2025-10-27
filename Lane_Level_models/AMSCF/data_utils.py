# -*- coding: utf-8 -*-
# data_utils.py — 数据加载、邻接清洗、相关图构建、归一化（按节点）
import os
import numpy as np
import pandas as pd
import torch


def load_npz_split(root: str):
    tr = np.load(os.path.join(root, "train.npz"))
    va = np.load(os.path.join(root, "val.npz"))
    te = np.load(os.path.join(root, "test.npz"))
    return (tr["x"], tr["y"]), (va["x"], va["y"]), (te["x"], te["y"])


# ---------- 关键：按节点的 z-score ----------
def nodewise_zscore_fit(x_tr: np.ndarray):
    # x_tr: (num, Tin, N, 1)  ->  (num*Tin, N)
    num, Tin, N, _ = x_tr.shape
    seq = x_tr.reshape(num * Tin, N)
    mu = seq.mean(axis=0)                         # (N,)
    std = seq.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)         # 防止除零
    return mu.astype(np.float32), std.astype(np.float32)

def apply_nodewise_zscore(arr: np.ndarray, mu: np.ndarray, std: np.ndarray):
    # arr: (..., N, 1)
    out = arr.copy().astype(np.float32)
    out = (out - mu.reshape(1, 1, -1, 1)) / std.reshape(1, 1, -1, 1)
    return out

def invert_nodewise_zscore(arr: np.ndarray, mu: np.ndarray, std: np.ndarray):
    return arr * std.reshape(1, 1, -1) + mu.reshape(1, 1, -1)


def load_adjacency(csv_path: str) -> np.ndarray:
    # 兼容：首行首列为索引/列名、左上角空
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    if pd.isna(df.iat[0, 0]):
        df2 = df.iloc[1:, 1:]
        if df2.shape[0] == df2.shape[1]:
            df = df2
    df = df.fillna(0.0)
    A = df.to_numpy(dtype=float)
    A[A < 0] = 0
    # 弱对称化 + 去自环 + 加单位阵
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    A = A + np.eye(A.shape[0], dtype=float)
    return A


def compute_corr_graph_from_all(x_train: np.ndarray, gamma: float = 1.5, keep_ratio: float = 0.25) -> np.ndarray:
    """
    用全训练集估计相关图；指数强化 + 行 top-k 稀疏化 + 对称化 + 自环
    x_train: (num, Tin, N, 1)
    """
    num, Tin, N, _ = x_train.shape
    seq = x_train.reshape(num * Tin, N)
    seq = (seq - seq.mean(0, keepdims=True)) / (seq.std(0, keepdims=True) + 1e-6)
    corr = np.corrcoef(seq.T)                                  # (N,N)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    corr = np.exp(gamma * corr) - 1.0                          # 强化强相关
    # 行 top-k 稀疏化
    k = max(1, int(N * keep_ratio))
    idx = np.argsort(-corr, axis=1)[:, :k]
    mask = np.zeros_like(corr)
    for i in range(N):
        mask[i, idx[i]] = 1.0
    corr = corr * mask
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, corr.diagonal() + 1.0)
    return corr


def density_proxy_from_flow(flow_train: np.ndarray, win: int = 5) -> np.ndarray:
    """
    若无密度通道，用滑动平均的流量作为密度代理，仅用于构建第二源相关图。
    """
    x = flow_train
    num, Tin, N, _ = x.shape
    seq = x.reshape(num * Tin, N)
    pad = np.pad(seq, ((win-1, 0), (0, 0)), mode='edge')
    smooth = np.vstack([pad[i:i+win, :].mean(axis=0) for i in range(seq.shape[0])])
    return smooth.reshape(num, Tin, N, 1)


def to_tensor(m, device):
    return torch.tensor(m, dtype=torch.float32, device=device)
