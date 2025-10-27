# input_data.py — 统一从你的 PeMS 目录加载 train/val/test 和邻接
import os
import numpy as np
import pandas as pd

# 固定你的数据目录
DATA_DIR =


def load_pems_npz(data_dir=DATA_DIR):
    """
    加载 PeMS 三个 npz（train/val/test）与 PEMS_adj.csv。
    返回：
      (x_train, y_train), (x_val, y_val), (x_test, y_test), adj
    其中：
      x_*: (B, T_in, N)  —— 已去掉末尾通道维
      y_*: (B, H,   N)
      adj: (N, N)
    """
    train = np.load(os.path.join(data_dir, "train.npz"))
    val   = np.load(os.path.join(data_dir, "val.npz"))
    test  = np.load(os.path.join(data_dir, "test.npz"))

    # squeeze 通道维 (B, T, N, 1) -> (B, T, N)
    x_train = np.squeeze(train["x"], axis=-1).astype(np.float32)
    x_val   = np.squeeze(val["x"],   axis=-1).astype(np.float32)
    x_test  = np.squeeze(test["x"],  axis=-1).astype(np.float32)

    y_train = np.squeeze(train["y"], axis=-1).astype(np.float32)
    y_val   = np.squeeze(val["y"],   axis=-1).astype(np.float32)
    y_test  = np.squeeze(test["y"],  axis=-1).astype(np.float32)

    # 读取邻接（首行首列是索引，数据从数据区读取即可）
    adj = pd.read_csv(os.path.join(data_dir, "PEMSF_adj.csv"), index_col=0).values.astype(np.float32)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), adj


# —— 以下旧数据接口保留空壳/简单实现，避免其他模块import时报错 —— #
def load_sz_data(dataset=None):
    raise NotImplementedError("This project uses PeMS NPZ only. Use load_pems_npz().")

def load_los_data(dataset=None):
    raise NotImplementedError("This project uses PeMS NPZ only. Use load_pems_npz().")

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    raise NotImplementedError("This project uses prepared NPZ splits. Not needed.")

def preprocess_data2(data, time_len, rate, seq_len, pre_len):
    raise NotImplementedError("This project uses prepared NPZ splits. Not needed.")
