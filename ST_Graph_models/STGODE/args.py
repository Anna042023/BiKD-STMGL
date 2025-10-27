import argparse

parser = argparse.ArgumentParser()

# -------------------- 基本训练参数 --------------------
parser.add_argument('--remote', action='store_true', help='the code run on a server')
parser.add_argument('--num-gpu', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')

# -------------------- 数据集设置 --------------------
# 这里改为你自己的数据集名称
parser.add_argument('--filename', type=str, default='pems_lane')

# 数据划分比例
parser.add_argument('--train-ratio', type=float, default=0.6, help='training set ratio')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='validation set ratio')

# 历史输入长度 & 预测长度（会被 run_stode.py 循环覆盖）
parser.add_argument('--his-length', type=int, default=12, help='input sequence length')
parser.add_argument('--pred-length', type=int, default=12, help='prediction sequence length')

# -------------------- 图参数 --------------------
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='threshold for semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='threshold for spatial matrix')

# -------------------- 优化器参数 --------------------
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

# -------------------- 其他选项 --------------------
parser.add_argument('--log', action='store_true', help='if write log to files')

args = parser.parse_args()
