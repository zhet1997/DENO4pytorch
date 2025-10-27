import os
import sys
import argparse
import time
import json
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# 路径注入，支持通过绝对路径运行本脚本
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)


# 数据与工具
from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision


def _select_device(device_arg: str, cuda_index: int) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_index}")
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    return device


def _split_dataset(inputs: np.ndarray,
                   outputs: np.ndarray,
                   ntrain: Optional[int],
                   nvalid: Optional[int]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    将完整数据按训练脚本约定划分：若未指定，则按 9:1 划分；否则前取训练，后取验证。
    返回 ((train_x, train_y), (valid_x, valid_y))
    """
    N = inputs.shape[0]
    if ntrain is None or nvalid is None:
        ntrain = int(N * 0.9)
        nvalid = N - ntrain
    else:
        assert ntrain + nvalid <= N, f"ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}"

    train_x = inputs[:ntrain]
    train_y = outputs[:ntrain]
    valid_x = inputs[N - nvalid:]
    valid_y = outputs[N - nvalid:]
    
    x_normalizer = DataNormer(train_x, method='mean-std')
    y_normalizer = DataNormer(train_y, method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    
    return (train_x, train_y), (valid_x, valid_y)


def _build_fno(args, device):
    from fno.FNOs import FNO2d
    from Demo.satellite_2d_base.run_FNO_satellite import feature_transform as fno_feature_transform
    modes = (args.modes_x, args.modes_y)
    net = FNO2d(in_dim=6, out_dim=1, modes=modes, width=args.width, depth=args.depth,
                steps=args.steps, padding=args.padding, activation='gelu').to(device)
    return net, fno_feature_transform


def _build_unet(args, device):
    from cnn.ConvNets import UNet2d
    from Demo.satellite_2d_base.run_UNet_satellite import feature_transform as unet_feature_transform
    # 输入/输出尺寸由数据动态决定，推理时不强绑定尺寸；UNet2d 构造需要参考训练脚本参数
    net = UNet2d(
        in_sizes=(256, 256, 6),
        out_sizes=(256, 256, 1),
        width=args.width,
        depth=args.depth,
        steps=args.steps,
        activation='gelu',
        dropout=args.dropout,
    ).to(device)
    return net, unet_feature_transform



# def _build_transformer(args, device):
#     from transformer.Transformers import FourierTransformer
#     from Demo.satellite_2d_base.run_Trans_satellite import feature_transform as trans_feature_transform
#     cfg = _maybe_load_yaml_config(args)
#     if cfg is None:
#         cfg = dict(
#             node_feats=6,
#             n_targets=1,
#             n_hidden=args.hidden,
#             n_head=args.nhead,
#             num_encoder_layers=args.enc_layers,
#             dim_feedforward=(2 * args.hidden) if args.ffn is None else args.ffn,
#             attention_type='fourier',
#             feat_extract_type='identity',
#             num_feat_layers=0,
#             graph_activation=False,
#             raw_laplacian=True,
#             pos_dim=2,
#             edge_feats=0,
#             layer_norm=True,
#             attn_norm=False,
#             batch_norm=False,
#             spacial_residual=False,
#             return_attn_weight=False,
#             seq_len=None,
#             bulk_regression=False,
#             decoder_type=args.decoder,
#             num_regressor_layers=2,
#             fourier_modes=args.fourier_modes,
#             freq_dim=64,
#             spacial_dim=2,
#             spacial_fc=True,
#             dropout=args.dropout,
#             xavier_init=1e-4,
#             diagonal_weight=1e-2,
#             symmetric_init=False,
#             debug=False,
#         )
#     net = FourierTransformer(**cfg).to(device)
#     return net, trans_feature_transform


# def _build_mlp(args, device):
#     from Demo.satellite_2d_base.run_MLP_satellite import MLP as MLPNet
#     s = 256 // args.down
#     in_dim = s * s * 6
#     out_dim = s * s * 1
#     layers = [in_dim]
#     for _ in range(max(args.layers - 2, 1)):
#         layers.append(args.hidden)
#     layers.append(out_dim)
#     net = MLPNet(layer_mat=layers, is_BatchNorm=False).to(device)
#     return net, None


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'net_model' in state:
        state_dict = state['net_model']
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=False)
    
    return model


def _compute_metrics(true_arr: np.ndarray, pred_arr: np.ndarray) -> Dict[str, float]:
    """
    输入形状：
    - true_arr: (N, H, W, 1)
    - pred_arr: (N, H, W, 1)
    返回整体平均 MSE / MAE 与 R2（按样本平均）。
    """
    assert true_arr.shape == pred_arr.shape
    N = true_arr.shape[0]
    eps = 1e-12

    per_sample_mse = ((true_arr - pred_arr) ** 2).reshape(N, -1).mean(axis=1)
    per_sample_mae = (np.abs(true_arr - pred_arr)).reshape(N, -1).mean(axis=1)

    # R2: 1 - SS_res/SS_tot（逐样本）
    true_flat = true_arr.reshape(N, -1)
    pred_flat = pred_arr.reshape(N, -1)
    true_mean = true_flat.mean(axis=1, keepdims=True)
    ss_res = ((true_flat - pred_flat) ** 2).sum(axis=1)
    ss_tot = ((true_flat - true_mean) ** 2).sum(axis=1) + eps
    per_sample_r2 = 1.0 - (ss_res / ss_tot)

    return {
        'avg_mse': float(per_sample_mse.mean()),
        'avg_mae': float(per_sample_mae.mean()),
        'avg_r2': float(per_sample_r2.mean()),
    }


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='卫星2D模型推理评估脚本（统一评估FNO/UNet/Transformer/MLP）')
    # 基本参数
    parser.add_argument('--model', type=str, required=True, choices=['fno', 'unet', 'trans', 'mlp'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=os.path.join('work_satellite'))
    parser.add_argument('--max_plots', type=int, default=200)
    # 划分与子集
    parser.add_argument('--ntrain', type=int, default=None)
    parser.add_argument('--nvalid', type=int, default=None)
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'all'])
    # 下采样（Transformer/MLP用）
    parser.add_argument('--down', type=int, default=8)
    # FNO 结构
    parser.add_argument('--modes_x', type=int, default=4)
    parser.add_argument('--modes_y', type=int, default=4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--padding', type=int, default=8)
    # UNet 结构
    parser.add_argument('--dropout', type=float, default=0.0)
    # Transformer 结构
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--ffn', type=int, default=None)
    parser.add_argument('--decoder', type=str, default='pointwise', choices=['pointwise', 'ifft', 'attention'])
    parser.add_argument('--fourier_modes', type=int, default=16)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--use_config', action='store_true', default=False)
    # MLP 结构
    parser.add_argument('--layers', type=int, default=4)

    args = parser.parse_args()

    # 保存目录
    ts = time.strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(args.save_dir, f"eval_{args.model}_{ts}")
    samples_dir = os.path.join(eval_dir, 'samples')
    _ensure_dir(eval_dir)
    _ensure_dir(samples_dir)

    # 设备
    device = _select_device(args.device, args.cuda_index)
    print(f"设备: {device}")

    # 数据加载与划分
    inputs, outputs = load_satellite_data(args.data_path)
    (train_x_np, train_y_np), (valid_x_np, valid_y_np) = _split_dataset(inputs, outputs, args.ntrain, args.nvalid)

    # 选择评估子集
    if args.model in ('trans', 'mlp'):
        assert 256 % args.down == 0, 'down 必须整除 256'

    if args.split == 'train':
        eval_x_np = train_x_np
        eval_y_np = train_y_np
    elif args.split == 'valid':
        eval_x_np = valid_x_np
        eval_y_np = valid_y_np
    else:  # all
        eval_x_np = np.concatenate([train_x_np, valid_x_np], axis=0)
        eval_y_np = np.concatenate([train_y_np, valid_y_np], axis=0)

    # 归一化（基于训练集统计，与训练范式一致）
    # 不同模型的数据形态在此分支处理
    if args.model in ('fno', 'unet', 'trans'):
        eval_x_t = torch.tensor(eval_x_np, dtype=torch.float32)
        eval_y_t = torch.tensor(eval_y_np, dtype=torch.float32)

        eval_ds = TensorDataset(eval_x_t, eval_y_t)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                 drop_last=False, num_workers=args.num_workers)
    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # 构建模型与特征网格函数
    if args.model == 'fno':
        net, feat_fn = _build_fno(args, device)
    elif args.model == 'unet':
        net, feat_fn = _build_unet(args, device)
    elif args.model == 'trans':
        net, feat_fn = _build_transformer(args, device)
    elif args.model == 'mlp':
        net, feat_fn = _build_mlp(args, device)
    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # 加载权重
    net = _load_checkpoint(net, args.ckpt, device)
    net.eval()

    # 推理与收集结果（反归一化以回到物理尺度）
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for bx, by in eval_loader:
            bx = bx.to(device)
            if args.model in ('fno', 'unet', 'trans'):
                gd = feat_fn(bx)
                pred = net(bx, gd)
                pred_np = pred.cpu().numpy()
                true_np = by.cpu().numpy()
                # 统一为 (N,H,W,1)
                preds_list.append(pred_np)
                trues_list.append(true_np)

    preds_all = np.concatenate(preds_list, axis=0)
    trues_all = np.concatenate(trues_list, axis=0)

    # 计算指标
    metrics = _compute_metrics(trues_all, preds_all)
    metrics.update({
        'dataset_size': int(trues_all.shape[0]),
        'split': args.split,
        'height': int(trues_all.shape[1]),
        'width': int(trues_all.shape[2]),
    })

    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(eval_dir, 'metrics.txt'), 'w') as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 可视化：复用 MatplotlibVision
    vision = MatplotlibVision(log_dir=eval_dir, input_name=('x',), field_name=('f',))

    # 保存对比图（truth/pred/error 三列），控制最大张数
    max_plots = 10
    num_to_plot = min(max_plots, trues_all.shape[0])
    for idx in range(num_to_plot):
        real = trues_all[idx]  # (H,W,1)
        pred = preds_all[idx]  # (H,W,1)

        # MatplotlibVision.plot_fields_grid 期望 (H,W,C)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        # 该方法内部会计算 fmin/fmax 并绘制 truth/pred/error
        vision.plot_fields_ms(fig, axs, real, pred, None, titles=['truth', 'predicted', 'error'])
        out_path = os.path.join(samples_dir, f"{idx:06d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # 记录配置
    cfg_dump = {
        'args': vars(args),
        'device': str(device),
        'metrics_path': os.path.join(eval_dir, 'metrics.json'),
        'samples_dir': samples_dir,
    }
    with open(os.path.join(eval_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    print(f"评估完成。输出目录: {eval_dir}")


if __name__ == '__main__':
    main()


