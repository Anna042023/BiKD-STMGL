import argparse
import os

def arguments():
    parser = argparse.ArgumentParser()

    # 设备 & 数据根路径
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default=r"/data/ZhangChao/wanganna")
    parser.add_argument('--dataset_name', type=str, choices=['PeMS', 'HuaNan', 'PeMSF'], default="PeMS")

    # 输入/输出
    parser.add_argument('--seq_len', type=int, default=12, help='输入历史步数，与你的 npz 一致（12）')
    parser.add_argument('--diffusion_step', type=int, default=0, help='占位，无用')
    parser.add_argument('--cheb_k', type=int, default=3, help='Chebyshev 阶数（用 I,A,A^2 近似）')

    # 模型宽度
    parser.add_argument('--hidden_dim', type=int, default=64)

    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=0.003)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--steps', type=eval, default=[40, 80])
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=20)

    # 节点与邻接矩阵路径（启动后会按数据集覆盖）
    parser.add_argument('--lane_num_nodes', type=int, default=40)
    parser.add_argument('--adj_path', type=str, default='')

    args = parser.parse_args()

    if args.dataset_name == "PeMS":
        args.lane_num_nodes = 40
        args.adj_path = os.path.join(args.dataset_root, "PeMS", "PEMS_adj.csv")
    elif args.dataset_name == "HuaNan":
        args.lane_num_nodes = 72
        args.adj_path = os.path.join(args.dataset_root, "HuaNan", "Huanan_adj.csv")
    elif args.dataset_name == "PeMSF":
        args.lane_num_nodes = 43
        args.adj_path = os.path.join(args.dataset_root, "PeMSF", "PEMSF_adj.csv")

    args.dataset_dir = os.path.join(args.dataset_root, args.dataset_name)
    return args
