import os
import sys
import argparse
import time
import json
import yaml
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
    返回 ((train_x, train_y), (valid_x, valid_y))，不进行归一化
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
    
    return (train_x, train_y), (valid_x, valid_y)


def _get_default_model_config(model_type: str) -> Dict[str, Any]:
    """返回模型默认配置参数"""
    if model_type == 'fno':
        return {
            'modes': (10, 10),
            'width': 64,
            'depth': 3,
            'steps': 1,
            'padding': 8,
        }
    elif model_type == 'mlp':
        return {
            'down': 4,
            'hidden': 1024,
            'layers': 5,
            's':64,
        }
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def _load_normalizers(ckpt_dir: str, train_x: np.ndarray, train_y: np.ndarray):
    """
    从 normalizers.yaml 加载归一化器，不存在则基于数据重新计算
    
    注意：YAML 中保存的是按通道的统计量（形状为 (C,)），需要 reshape 为 (1,1,C) 以支持广播
    """
    yaml_path = os.path.join(ckpt_dir, 'normalizers.yaml')
    if os.path.exists(yaml_path):
        print(f"从 {yaml_path} 加载归一化器")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # 加载统计量：YAML 中保存的是每个通道的统计量 (C,)
        # 需要 reshape 为 (1, 1, C) 以便与 (N, H, W, C) 格式广播
        x_mean = np.array(cfg['x_mean'], dtype=np.float32).reshape(1, 1, -1)  # (1,1,6)
        x_std = np.array(cfg['x_std'], dtype=np.float32).reshape(1, 1, -1)
        y_mean = np.array(cfg['y_mean'], dtype=np.float32).reshape(1, 1, -1)  # (1,1,1)
        y_std = np.array(cfg['y_std'], dtype=np.float32).reshape(1, 1, -1)
        
        # 创建归一化器并手动设置统计量
        x_normalizer = DataNormer(train_x, method=cfg['method'])
        x_normalizer.mean = x_mean
        x_normalizer.std = x_std
        
        y_normalizer = DataNormer(train_y, method=cfg['method'])
        y_normalizer.mean = y_mean
        y_normalizer.std = y_std
    else:
        print("未找到 normalizers.yaml，基于数据重新计算归一化器")
        x_normalizer = DataNormer(train_x, method='mean-std')
        y_normalizer = DataNormer(train_y, method='mean-std')
    
    return x_normalizer, y_normalizer


def feature_transform(x):
    """FNO 使用的网格特征变换"""
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


def _build_fno(device):
    from fno.FNOs import FNO2d
    config = _get_default_model_config('fno')
    net = FNO2d(
        in_dim=6, 
        out_dim=1, 
        modes=config['modes'], 
        width=config['width'], 
        depth=config['depth'],
        steps=config['steps'], 
        padding=config['padding'], 
        activation='gelu'
    ).to(device)
    return net


def _build_mlp(device, ckpt_dir: str):
    """
    构建 MLP 模型，从检查点自动推断参数
    """
    from Demo.satellite_2d_dssl.run_MLP_satellite_base_new import MLP
    
    # 从检查点推断实际参数
    config = _get_default_model_config('mlp')
    
    s = config['s']
    in_dim = s*s*6
    out_dim = s*s*1
    hidden = config['hidden']
    num_layers = config['layers']
    
    # 构建层列表
    layers = [in_dim]
    for _ in range(num_layers - 2):
        layers.append(hidden)
    layers.append(out_dim)
    
    print(f"构建 MLP 网络: {layers}")
    
    net = MLP(
        layer_mat=layers, 
        is_BatchNorm=False,
        input_shape_2d=(s, s, 6), 
        output_shape_2d=(s, s, 1)
    ).to(device)
    return net, config['down']


def _load_checkpoint(model: torch.nn.Module, ckpt_dir: str, device: torch.device) -> torch.nn.Module:
    """从文件夹加载 latest_model.pth 权重"""
    ckpt_path = os.path.join(ckpt_dir, 'latest_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到检查点文件: {ckpt_path}")
    
    print(f"加载检查点: {ckpt_path}")
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
    parser = argparse.ArgumentParser(description='卫星2D模型推理评估脚本（FNO/MLP）')
    # 基本参数
    parser.add_argument('--model', type=str, default='fno', choices=['fno', 'mlp'])
    parser.add_argument('--data_path', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5')
    parser.add_argument('--ckpt', type=str, required=True, help='检查点文件夹路径')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='work_satellite_eval')
    parser.add_argument('--max_plots', type=int, default=10)
    # 划分与子集
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--nvalid', type=int, default=500)
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'all'])

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

    # 数据加载与划分（原始物理空间）
    inputs, outputs = load_satellite_data(args.data_path)
    (train_x_np, train_y_np), (valid_x_np, valid_y_np) = _split_dataset(inputs, outputs, args.ntrain, args.nvalid)

    # 从检查点文件夹加载归一化器
    x_normalizer, y_normalizer = _load_normalizers(args.ckpt, train_x_np, train_y_np)

    # 选择评估子集
    if args.split == 'train':
        eval_x_np = train_x_np
        eval_y_np = train_y_np
    elif args.split == 'valid':
        eval_x_np = valid_x_np
        eval_y_np = valid_y_np
    else:  # all
        eval_x_np = np.concatenate([train_x_np, valid_x_np], axis=0)
        eval_y_np = np.concatenate([train_y_np, valid_y_np], axis=0)

    # 构建模型（MLP 需要先构建以获取 down 参数）
    if args.model == 'fno':
        net = _build_fno(device)
        model_down = None  # FNO 不需要下采样
    elif args.model == 'mlp':
        net, model_down = _build_mlp(device, args.ckpt)
    else:
        raise ValueError(f"未知模型类型: {args.model}")
    
    # 根据模型类型处理数据
    if args.model == 'fno':
        # FNO: 使用64x64下采样数据（与训练时保持一致）
        down = 4
        s = 256 // down
        eval_x_down = eval_x_np[:, ::down, ::down, :]  # (N, s, s, 6)
        eval_y_down = eval_y_np[:, ::down, ::down, :] 
        eval_x_norm = x_normalizer.norm(eval_x_down)
        eval_y_norm = y_normalizer.norm(eval_y_down)
        eval_x_t = torch.tensor(eval_x_norm, dtype=torch.float32)
        eval_y_t = torch.tensor(eval_y_norm, dtype=torch.float32)
        
    elif args.model == 'mlp':
        # MLP: 下采样并展平（使用从检查点推断的 down）
        down = 4
        s = 256 // down
        eval_x_down = eval_x_np[:, ::down, ::down, :]  # (N, s, s, 6)
        eval_y_down = eval_y_np[:, ::down, ::down, :]  # (N, s, s, 1)
        
        eval_x_norm = x_normalizer.norm(eval_x_down)
        eval_y_norm = y_normalizer.norm(eval_y_down)
        
        # 展平为 1D
        eval_x_t = torch.tensor(eval_x_norm.reshape(eval_x_norm.shape[0], -1), dtype=torch.float32)
        eval_y_t = torch.tensor(eval_y_norm.reshape(eval_y_norm.shape[0], -1), dtype=torch.float32)
    else:
        raise ValueError(f"未知模型类型: {args.model}")
    
    eval_ds = TensorDataset(eval_x_t, eval_y_t)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.num_workers)

    # 加载权重
    net = _load_checkpoint(net, args.ckpt, device)
    net.eval()

    # 推理与收集结果
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for bx, by in eval_loader:
            bx = bx.to(device)
            
            if args.model == 'fno':
                gd = feature_transform(bx)
                pred = net(bx, gd)
            elif args.model == 'mlp':
                pred = net(bx)
            
            preds_list.append(pred.cpu().numpy())
            trues_list.append(by.cpu().numpy())

    preds_all = np.concatenate(preds_list, axis=0)
    trues_all = np.concatenate(trues_list, axis=0)
    
    # 反归一化到物理空间
    if args.model == 'mlp':
        # MLP 输出需要 reshape 回 2D
        down = model_down
        s = 256 // down
        preds_all = preds_all.reshape((-1, s, s, 1))
        trues_all = trues_all.reshape((-1, s, s, 1))
    
    preds_all = y_normalizer.back(preds_all)
    trues_all = y_normalizer.back(trues_all)

    # 保存预测和真实值的numpy数组
    if args.split == 'train':
        np.save(os.path.join(eval_dir, 'train_pred.npy'), preds_all)
        np.save(os.path.join(eval_dir, 'train_true.npy'), trues_all)
        print(f"已保存: train_pred.npy 和 train_true.npy，形状: {preds_all.shape}")
    elif args.split == 'valid':
        np.save(os.path.join(eval_dir, 'valid_pred.npy'), preds_all)
        np.save(os.path.join(eval_dir, 'valid_true.npy'), trues_all)
        print(f"已保存: valid_pred.npy 和 valid_true.npy，形状: {preds_all.shape}")
    else:  # all
        # 需要分开保存train和valid数据
        train_size = len(train_x_np)
        valid_size = len(valid_x_np)
        
        train_preds = preds_all[:train_size]
        train_trues = trues_all[:train_size]
        valid_preds = preds_all[train_size:]
        valid_trues = trues_all[train_size:]
        
        np.save(os.path.join(eval_dir, 'train_pred.npy'), train_preds)
        np.save(os.path.join(eval_dir, 'train_true.npy'), train_trues)
        np.save(os.path.join(eval_dir, 'valid_pred.npy'), valid_preds)
        np.save(os.path.join(eval_dir, 'valid_true.npy'), valid_trues)
        print(f"已保存: train_pred.npy 和 train_true.npy，形状: {train_preds.shape}")
        print(f"已保存: valid_pred.npy 和 valid_true.npy，形状: {valid_preds.shape}")

    # 计算指标
    metrics = _compute_metrics(trues_all, preds_all)
    metrics.update({
        'dataset_size': int(trues_all.shape[0]),
        'split': args.split,
        'height': int(trues_all.shape[1]),
        'width': int(trues_all.shape[2]),
    })

    with open(os.path.join(eval_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(eval_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 打印指标
    print("\n评估指标:")
    print(f"  平均 MSE: {metrics['avg_mse']:.6e}")
    print(f"  平均 MAE: {metrics['avg_mae']:.6e}")
    print(f"  平均 R²: {metrics['avg_r2']:.6f}")
    
    # 可视化：复用 MatplotlibVision
    vision = MatplotlibVision(log_dir=eval_dir, input_name=('x',), field_name=('f',))

    # 保存对比图（truth/pred/error 三列），控制最大张数
    num_to_plot = min(args.max_plots, trues_all.shape[0])
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
    with open(os.path.join(eval_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    print(f"评估完成。输出目录: {eval_dir}")


if __name__ == '__main__':
    main()


