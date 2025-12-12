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
    elif model_type == 'trans':
        return {
            'down': 8,
            'hidden': 96,
            'nhead': 2,
            'enc_layers': 2,
            'ffn': None,
            'decoder': 'pointwise',
            'fourier_modes': 16,
        }
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def _load_normalizers(ckpt_dir: str, train_x: np.ndarray, train_y: np.ndarray, model_type: str = 'fno'):
    """
    从 normalizers.yaml 加载归一化器，不存在则基于数据重新计算
    
    参数:
        model_type: 'fno' 或 'mlp'，用于确定归一化器统计量的形状
            - FNO: 统计量形状为 (6,) 或 (1,)，需要 reshape 为 (1,1,C) 以支持 (N,H,W,C) 广播
            - MLP: 统计量形状为 (24576,) 或 (4096,)，用于展平后的数据 (N, features)
    """
    yaml_path = os.path.join(ckpt_dir, 'normalizers.yaml')
    if os.path.exists(yaml_path):
        print(f"从 {yaml_path} 加载归一化器")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        x_mean = np.array(cfg['x_mean'], dtype=np.float32)
        x_std = np.array(cfg['x_std'], dtype=np.float32)
        y_mean = np.array(cfg['y_mean'], dtype=np.float32)
        y_std = np.array(cfg['y_std'], dtype=np.float32)
        
        # 根据模型类型设置归一化器统计量的形状
        if model_type in ('fno', 'trans'):
            # FNO/Transformer: 统计量按通道保存 (C,)，需要 reshape 为 (1, 1, C) 以支持 (N, H, W, C) 广播
            x_mean = x_mean.reshape(1, 1, -1)  # (1,1,6)
            x_std = x_std.reshape(1, 1, -1)
            y_mean = y_mean.reshape(1, 1, -1)  # (1,1,1)
            y_std = y_std.reshape(1, 1, -1)
            # 创建归一化器并手动设置统计量
            x_normalizer = DataNormer(train_x, method=cfg['method'])
            x_normalizer.mean = x_mean
            x_normalizer.std = x_std
            y_normalizer = DataNormer(train_y, method=cfg['method'])
            y_normalizer.mean = y_mean
            y_normalizer.std = y_std
        elif model_type == 'mlp':
            # MLP: 统计量可能是按展平后的特征维度保存的 (features,)，也可能是按通道保存的 (C,)
            # 如果是从自监督训练脚本保存的，可能是按通道保存的 (6,) 或 (1,)
            # 需要根据实际形状判断并转换
            
            # 检查是否是按通道保存的（自监督训练脚本保存的格式）
            if x_mean.shape[0] == 6 and y_mean.shape[0] == 1:
                # 这是按通道保存的2D归一化器（自监督训练脚本保存的格式）
                # 训练时是先归一化2D数据再展平，所以评估时也应该使用2D归一化器
                # 将统计量reshape为 (1, 1, C) 以支持 (N, H, W, C) 广播
                print(f"检测到按通道保存的归一化器 (x_mean.shape={x_mean.shape}, y_mean.shape={y_mean.shape})")
                print(f"使用2D归一化器（训练时先归一化2D数据再展平）")
                
                x_mean = x_mean.reshape(1, 1, -1)  # (1,1,6)
                x_std = x_std.reshape(1, 1, -1)
                y_mean = y_mean.reshape(1, 1, -1)  # (1,1,1)
                y_std = y_std.reshape(1, 1, -1)
                
                # 创建归一化器并手动设置统计量（使用2D格式的训练数据初始化）
                x_normalizer = DataNormer(train_x, method=cfg['method'])
                x_normalizer.mean = x_mean
                x_normalizer.std = x_std
                y_normalizer = DataNormer(train_y, method=cfg['method'])
                y_normalizer.mean = y_mean
                y_normalizer.std = y_std
            else:
                # 已经是展平后的归一化器，直接使用
                dummy_x = np.zeros((1, x_mean.shape[0]), dtype=np.float32)
                dummy_y = np.zeros((1, y_mean.shape[0]), dtype=np.float32)
                x_normalizer = DataNormer(dummy_x, method=cfg['method'])
                x_normalizer.mean = x_mean  # 保持 (features,) 形状
                x_normalizer.std = x_std
                y_normalizer = DataNormer(dummy_y, method=cfg['method'])
                y_normalizer.mean = y_mean
                y_normalizer.std = y_std
        else:
            raise ValueError(f"未知模型类型: {model_type}")
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
    from Demo.satellite_2d_base.run_MLP_satellite import MLP
    
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
        is_BatchNorm=False
    ).to(device)
    return net, config['down']


def _build_trans(device, ckpt_dir: str):
    """
    构建 Transformer 模型，尝试从检查点目录加载配置文件，否则使用默认配置
    """
    from transformer.Transformers import FourierTransformer
    from Demo.satellite_2d_base.utils import load_yaml_config
    
    # 尝试从检查点目录加载配置文件
    config_path = '/data/wqn/DENO4pytorch/data/configs/transformer_config_sate.yml'
    if not os.path.exists(config_path):
        # 尝试项目根目录的默认配置文件
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
        default_config_path = os.path.join(PROJECT_ROOT, 'configs', 'transformer_config_sate.yml')
        if os.path.exists(default_config_path):
            config_path = default_config_path
    
    # 获取默认配置
    config = _get_default_model_config('trans')
    down = config['down']
    s = 256 // down
    
    # 如果找到配置文件，加载它
    if os.path.exists(config_path):
        print(f"从 {config_path} 加载 Transformer 配置")
        raw_cfg = load_yaml_config(config_path) or {}
        # 兼容顶层名称
        if isinstance(raw_cfg, dict) and len(raw_cfg) == 1:
            cfg = list(raw_cfg.values())[0]
        elif isinstance(raw_cfg, dict):
            cfg = raw_cfg.get('PakB_2d', raw_cfg)
        else:
            cfg = {}
        
        # 强制覆盖关键字段
        cfg['node_feats'] = 6
        cfg['n_targets'] = 1
        cfg['pos_dim'] = 2
        cfg['spacial_dim'] = 2
        if cfg.get('feat_extract_type', None) in [None, 'None']:
            cfg['feat_extract_type'] = 'identity'
        
        # 根据下采样分辨率自动收紧 fourier_modes
        if 'fourier_modes' in cfg:
            safe_modes = max(1, min(int(cfg['fourier_modes']), int(s // 2 + 1)))
            cfg['fourier_modes'] = safe_modes
    else:
        # 使用默认配置
        print("使用默认 Transformer 配置")
        cfg = dict(
            node_feats=6,
            n_targets=1,
            n_hidden=config['hidden'],
            n_head=config['nhead'],
            num_encoder_layers=config['enc_layers'],
            dim_feedforward=(2 * config['hidden']) if config['ffn'] is None else config['ffn'],
            attention_type='fourier',
            feat_extract_type='identity',
            num_feat_layers=0,
            graph_activation=False,
            raw_laplacian=True,
            pos_dim=2,
            edge_feats=0,
            layer_norm=True,
            attn_norm=False,
            batch_norm=False,
            spacial_residual=False,
            return_attn_weight=False,
            seq_len=None,
            bulk_regression=False,
            decoder_type=config['decoder'],
            num_regressor_layers=2,
            fourier_modes=min(config['fourier_modes'], s // 2 + 1),
            freq_dim=64,
            spacial_dim=2,
            spacial_fc=True,
            dropout=0.0,
            xavier_init=1e-4,
            diagonal_weight=1e-2,
            symmetric_init=False,
            debug=False,
        )
    
    print(f"构建 Transformer 网络: down={down}, s={s}, fourier_modes={cfg.get('fourier_modes', 'N/A')}")
    
    net = FourierTransformer(**cfg).to(device)
    return net, down


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
    parser = argparse.ArgumentParser(description='卫星2D模型推理评估脚本（FNO/MLP/Transformer）')
    # 基本参数
    parser.add_argument('--model', type=str, default='fno', choices=['fno', 'mlp', 'trans'])
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

    # 从检查点文件夹加载归一化器（需要根据模型类型处理）
    x_normalizer, y_normalizer = _load_normalizers(args.ckpt, train_x_np, train_y_np, model_type=args.model)

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

    # 构建模型（MLP 和 Trans 需要先构建以获取 down 参数）
    if args.model == 'fno':
        net = _build_fno(device)
        model_down = None  # FNO 不需要下采样
    elif args.model == 'mlp':
        net, model_down = _build_mlp(device, args.ckpt)
    elif args.model == 'trans':
        net, model_down = _build_trans(device, args.ckpt)
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
        
    elif args.model == 'trans':
        # Transformer: 使用32x32下采样数据（与训练时保持一致，down=8）
        down = model_down if model_down else 8
        s = 256 // down
        eval_x_down = eval_x_np[:, ::down, ::down, :]  # (N, s, s, 6)
        eval_y_down = eval_y_np[:, ::down, ::down, :]
        eval_x_norm = x_normalizer.norm(eval_x_down)
        eval_y_norm = y_normalizer.norm(eval_y_down)
        eval_x_t = torch.tensor(eval_x_norm, dtype=torch.float32)
        eval_y_t = torch.tensor(eval_y_norm, dtype=torch.float32)
        
    elif args.model == 'mlp':
        # MLP: 下采样、归一化（2D格式），然后展平（与训练时保持一致）
        # 训练时是先归一化2D数据再展平，所以评估时也应该先归一化2D数据再展平
        down = 4
        s = 256 // down
        eval_x_down = eval_x_np[:, ::down, ::down, :]  # (N, s, s, 6)
        eval_y_down = eval_y_np[:, ::down, ::down, :]  # (N, s, s, 1)
        
        # 先归一化2D数据（与训练时保持一致）
        eval_x_norm = x_normalizer.norm(eval_x_down)  # (N, s, s, 6)
        eval_y_norm = y_normalizer.norm(eval_y_down)  # (N, s, s, 1)
        
        # 然后展平为 (N, features)
        eval_x_flat = eval_x_norm.reshape(eval_x_norm.shape[0], -1)  # (N, s*s*6)
        eval_y_flat = eval_y_norm.reshape(eval_y_norm.shape[0], -1)  # (N, s*s*1)
        
        # 转换为tensor
        eval_x_t = torch.tensor(eval_x_flat, dtype=torch.float32)
        eval_y_t = torch.tensor(eval_y_flat, dtype=torch.float32)
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
            elif args.model == 'trans':
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
        # MLP: 反归一化时，需要先reshape回2D，然后反归一化（与训练时保持一致）
        down = model_down if model_down else 4
        s = 256 // down
        # 先reshape回2D
        preds_all = preds_all.reshape((-1, s, s, 1))  # (N, s, s, 1)
        trues_all = trues_all.reshape((-1, s, s, 1))  # (N, s, s, 1)
        # 然后反归一化（2D格式）
        preds_all = y_normalizer.back(preds_all)  # (N, s, s, 1)
        trues_all = y_normalizer.back(trues_all)  # (N, s, s, 1)
    else:
        # FNO 和 Transformer: 直接反归一化（已经是2D格式）
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


