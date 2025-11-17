import os
import sys
import argparse
import time
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 路径注入，支持绝对路径运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from collections import OrderedDict
from Utilizes.process_data import DataNormer
from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
from Demo.satellite_2d_dssl.dataset_selfsup import (
    load_selfsup_data,
    dimension_scaling_Tensor
)
from Demo.satellite_2d_base.utils import load_yaml_config
from Utilizes.visual_data import MatplotlibVision

class MLP(nn.Module):
    def __init__(self, layer_mat=None, is_BatchNorm=False, input_shape_2d=None, output_shape_2d=None):
        super().__init__()
        if layer_mat is None:
            raise ValueError("layer_mat must be provided")
        if input_shape_2d is None or output_shape_2d is None:
            raise ValueError("input_shape_2d and output_shape_2d must be provided")
        
        self.input_shape_2d = input_shape_2d  # (H, W, C_in)
        self.output_shape_2d = output_shape_2d  # (H, W, C_out)
        self.depth = len(layer_mat)
        activation = nn.GELU

        layer_list = []
        for i in range(self.depth - 2):
            layer_list.append((f'layer_{i}', nn.Linear(layer_mat[i], layer_mat[i + 1])))
            if is_BatchNorm:
                layer_list.append((f'batchnorm_{i}', nn.BatchNorm1d(layer_mat[i + 1])))
            layer_list.append((f'activation_{i}', activation()))
        layer_list.append((f'layer_{self.depth - 2}', nn.Linear(layer_mat[-2], layer_mat[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

        
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C_in) - 2D格式输入，或 (B, H*W*C_in) - 1D格式
        Returns:
            out: (B, H, W, C_out) - 2D格式输出，或 (B, H*W*C_out) - 1D格式
        """
        if x.dim() == 4:
            # 2D输入，需要展平
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
            out_flat = self.layers(x_flat)
            out = out_flat.reshape(B, *self.output_shape_2d)
            return out
        else:
            # 1D输入，直接处理
            return self.layers(x)



def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)

        pred = netmodel(xx)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)

            pred = netmodel(xx)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()
    return valid_loss / (batch + 1)


def build_input_dim_matrix_from_yaml_v2(yaml_path: str) -> np.ndarray:
    """
    从新版本 augmentation_satellite.yml 解析输入通道的量纲矩阵。
    
    新版本YAML结构：
    - channel_mapping: 通道名称到索引的映射
    - dim_exponents: 通道名称到量纲指数的映射
    
    返回形状为 (4, C) 的量纲矩阵，其中 C 为输入通道数（6）。
    """
    cfg = load_yaml_config(yaml_path)
    if not cfg:
        raise RuntimeError(f"无法加载 YAML 配置: {yaml_path}")
    channel_mapping = cfg.get('channel_mapping', {})
    dim_exponents = cfg.get('dim_exponents', {})
    # 确定通道数（假设为6）
    num_channels = 6
    dim_mat = np.zeros((4, num_channels), dtype=np.float32)
    # 遍历所有通道映射
    for channel_name, channel_idx in channel_mapping.items():
            dim_mat[:, channel_idx] = np.array(dim_exponents[channel_name], dtype=np.float32)
    return dim_mat


def _pick_random_alpha(alpha_batch: torch.Tensor) -> torch.Tensor:
    """
    从批次中为每个样本随机选择一组 alpha。
    alpha_batch: (B, K, 4)，B 为批次大小，K 为每样本可用系数组数
    return: (B, 4)
    """
    B, K, _ = alpha_batch.shape
    idx = torch.randint(low=0, high=K, size=(B,), device=alpha_batch.device)
    chosen = alpha_batch[torch.arange(B, device=alpha_batch.device), idx, :]
    return chosen


def train_selfsup(dataloader_ss,
                  netmodel,
                  device,
                  lossfunc,
                  x_normalizer: DataNormer,
                  y_normalizer: DataNormer,
                  input_dim_mat: torch.Tensor,
                  output_dim_mat: torch.Tensor,
                  optimizer_ss,
                  skip_backward: bool = False) -> float:
    """
    自监督训练：基于线性物理相似性（MLP版本）
    输入输出均为2D格式，MLP内部处理展平
    
    参数:
        skip_backward: 是否跳过反向传播（学习率为 0 时使用，节约算力）
    
    算力优化：
        - 当 skip_backward=True 时，使用 torch.no_grad() 跳过梯度计算
        - 仅计算损失值用于监控，不执行参数更新
    """
    netmodel.train()
    epoch_loss = 0.0
    
    if skip_backward:
        # 路径 1：跳过梯度计算（学习率为 0 时）
        with torch.no_grad():
            for batch, (xx, alpha_all) in enumerate(dataloader_ss):
                xx = xx.to(device=device, dtype=torch.float32)
                alpha_all = alpha_all.to(device=device, dtype=torch.float32)
                
                x_den_anc = x_normalizer.back(xx)
                alpha = _pick_random_alpha(alpha_all)
                x_den_far = dimension_scaling_Tensor(x_den_anc, bc_dim_mat=input_dim_mat, bc_dim_coef=alpha)
                x_norm_anc = xx
                x_norm_far = x_normalizer.norm(x_den_far)
                
                # MLP前向（2D输入）
                y_norm_anc = netmodel(x_norm_anc)
                y_norm_far = netmodel(x_norm_far)
                
                # 反归一化、物理反映射、再归一化
                y_den_far = y_normalizer.back(y_norm_far)
                y_den_far_back = dimension_scaling_Tensor(y_den_far, bc_dim_mat=output_dim_mat, bc_dim_coef=-alpha)
                y_norm_far_back = y_normalizer.norm(y_den_far_back)
                
                loss = lossfunc(y_norm_far_back, y_norm_anc)
                epoch_loss += float(loss.item())
    else:
        # 路径 2：正常训练（学习率 > 0 时）
        for batch, (xx, alpha_all) in enumerate(dataloader_ss):
            xx = xx.to(device=device, dtype=torch.float32)
            alpha_all = alpha_all.to(device=device, dtype=torch.float32)
            
            x_den_anc = x_normalizer.back(xx)
            alpha = _pick_random_alpha(alpha_all)
            x_den_far = dimension_scaling_Tensor(x_den_anc, bc_dim_mat=input_dim_mat, bc_dim_coef=alpha)
            x_norm_anc = xx
            x_norm_far = x_normalizer.norm(x_den_far)
            
            # MLP前向（2D输入）
            y_norm_anc = netmodel(x_norm_anc)
            y_norm_far = netmodel(x_norm_far)
            
            # 反归一化、物理反映射、再归一化
            y_den_far = y_normalizer.back(y_norm_far)
            y_den_far_back = dimension_scaling_Tensor(y_den_far, bc_dim_mat=output_dim_mat, bc_dim_coef=-alpha)
            y_norm_far_back = y_normalizer.norm(y_den_far_back)
            
            loss = lossfunc(y_norm_far_back, y_norm_anc)
            
            optimizer_ss.zero_grad()
            loss.backward()
            optimizer_ss.step()
            
            epoch_loss += float(loss.item())
    
    avg_loss = epoch_loss / (batch + 1)
    return avg_loss


def inference(dataloader, netmodel, device, down=4):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
        device: 设备
        down: 下采样倍数
    Returns:
        coords, grid, true_fields, pred_fields (所有形状为 N,H,W,C)
    """
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        pred = netmodel(xx)
    
    # 计算reshape后的空间尺寸
    s = 256 // down
    batch_size = xx.shape[0]
    
    # Reshape回2D格式: (B, s*s*C) -> (B, s, s, C)
    xx_reshaped = xx.cpu().numpy().reshape(batch_size, s, s, 6)
    yy_reshaped = yy.numpy().reshape(batch_size, s, s, 1)
    pred_reshaped = pred.cpu().numpy().reshape(batch_size, s, s, 1)
    
    # 生成虚拟的grid（为了保持接口一致）
    gridx = np.linspace(0, 1, s).reshape(1, s, 1, 1).repeat(batch_size, axis=0).repeat(s, axis=2)
    gridy = np.linspace(0, 1, s).reshape(1, 1, s, 1).repeat(batch_size, axis=0).repeat(s, axis=1)
    grid = np.concatenate([gridx, gridy], axis=-1)
    
    return xx_reshaped, grid, yy_reshaped, pred_reshaped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP 卫星热数据训练脚本')
    parser.add_argument('--data_path', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5')
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--nvalid', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=6)
    parser.add_argument('--down', type=int, default=4, help='空间下采样倍数，256/down，建议 4/8')
    parser.add_argument('--hidden', type=int, default=1024, help='MLP隐藏层宽度')
    parser.add_argument('--layers', type=int, default=4, help='总层数（含输入输出）最少3')
    parser.add_argument('--work_dir', type=str, default=os.path.join('work_satellite'))
    # 自监督参数
    parser.add_argument('--selfsup_dir', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c_sim1/')
    parser.add_argument('--self_batch_size', type=int, default=32)
    parser.add_argument('--self_lr_final', type=float, default=1e-8)
    parser.add_argument('--self_sample_limit', type=int, default=None)
    parser.add_argument('--noise_std', type=float, default=0.05, 
                       help='训练集输出噪声标准差（归一化空间）')
    parser.add_argument('--noise_type', type=str, default='independent', choices=['independent', 'correlated'])
    args = parser.parse_args()

    net_name = 'MLP_BASE'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_path = os.path.join(args.work_dir, f'{net_name}_noise{args.noise_std}_{timestamp}')
    os.makedirs(work_path, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(work_path, 'training.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f'工作路径: {work_path}')
    logger.info(f'训练样本数: {args.ntrain}, 验证样本数: {args.nvalid}')
    logger.info(f'输出噪声标准差（归一化空间）: {args.noise_std}')

    # 设备
    if torch.cuda.is_available():
        Device = torch.device(f'cuda:{args.cuda_index}')
    else:
        Device = torch.device('cpu')

    # 加载数据（inputs: N,256,256,6; outputs: N,256,256,1）
    inputs, outputs = load_satellite_data(args.data_path, noise_scale=args.noise_std, noise_type=args.noise_type)
    N = inputs.shape[0]

    ntrain = args.ntrain
    nvalid = args.nvalid
    assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'
    
    # 首先保持2D格式进行归一化（与自监督一致）
    s = 256 // args.down
    in_dim = s * s * 6
    out_dim = s * s * 1

    train_x_2d = torch.tensor(
        inputs[:ntrain][:, ::args.down, ::args.down, :],  # (N, s, s, 6)
        dtype=torch.float32
    )
    train_y_2d = torch.tensor(
        outputs[:ntrain][:, ::args.down, ::args.down, :],  # (N, s, s, 1)
        dtype=torch.float32
    )
    valid_x_2d = torch.tensor(
        inputs[N - nvalid:][:, ::args.down, ::args.down, :],
        dtype=torch.float32
    )
    valid_y_2d = torch.tensor(
        outputs[N - nvalid:][:, ::args.down, ::args.down, :],
        dtype=torch.float32
    )

    # 归一化（基于2D格式，与自监督一致）
    x_normalizer = DataNormer(train_x_2d.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y_2d.numpy(), method='mean-std')
    train_x_2d = x_normalizer.norm(train_x_2d)
    valid_x_2d = x_normalizer.norm(valid_x_2d)
    train_y_2d = y_normalizer.norm(train_y_2d)
    valid_y_2d = y_normalizer.norm(valid_y_2d)
    
    # 展平为1D用于监督训练
    train_x = train_x_2d.reshape(ntrain, in_dim)
    train_y = train_y_2d.reshape(ntrain, out_dim)
    valid_x = valid_x_2d.reshape(nvalid, in_dim)
    valid_y = valid_y_2d.reshape(nvalid, out_dim)
    
    # 保存归一化器信息
    normalizer_info = {
        'x_mean': x_normalizer.mean.tolist(),
        'x_std': x_normalizer.std.tolist(),
        'y_mean': y_normalizer.mean.tolist(),
        'y_std': y_normalizer.std.tolist(),
        'method': 'mean-std'
    }
    with open(os.path.join(work_path, 'normalizers.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(normalizer_info, f, allow_unicode=True)
    logger.info(f'归一化器已保存: x_mean.shape={x_normalizer.mean.shape}, y_mean.shape={y_normalizer.mean.shape}')

    # 创建训练集 DataLoader（带噪声支持）
    train_dataset = TensorDataset(train_x, train_y)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 自监督数据加载（保持2D格式）
    ss_inputs_np, ss_alphas_np = load_selfsup_data(args.selfsup_dir, sample_limit=args.self_sample_limit)
    logger.info(f"自监督数据: ss_inputs shape={ss_inputs_np.shape}, ss_alphas shape={ss_alphas_np.shape}")
    # 与监督相同下采样
    ss_inputs_np = ss_inputs_np[:, ::args.down, ::args.down, :].astype(np.float32)
    ss_alphas_np = ss_alphas_np.astype(np.float32)
    
    # 转为tensor并归一化（保持2D格式）
    ss_inputs_2d = torch.tensor(ss_inputs_np, dtype=torch.float32)
    ss_alphas_tensor = torch.tensor(ss_alphas_np, dtype=torch.float32)
    ss_inputs_2d_norm = x_normalizer.norm(ss_inputs_2d)
    
    # 创建自监督数据集（使用归一化后的2D tensor）
    ss_dataset = TensorDataset(ss_inputs_2d_norm, ss_alphas_tensor)
    ss_loader = DataLoader(ss_dataset, batch_size=args.self_batch_size, shuffle=True, drop_last=False)

    # 量纲矩阵：输入从新版本YAML，输出为temperature的量纲
    yaml_path = os.path.join(PROJECT_ROOT, 'Demo', 'satellite_2d_dssl', 'augmentation_satellite.yml')
    input_dim_mat_np = build_input_dim_matrix_from_yaml_v2(yaml_path)
    input_dim_mat = torch.tensor(input_dim_mat_np, dtype=torch.float32, device=Device)
    logger.info(f"输入量纲矩阵形状: {input_dim_mat.shape}")
    
    output_dim_mat = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32, device=Device)
    logger.info(f"输出量纲矩阵形状: {output_dim_mat.shape}")

    layers = [in_dim]
    for _ in range(max(args.layers - 2, 1)):
        layers.append(args.hidden)
    layers.append(out_dim)
    Net_model = MLP(layer_mat=layers, is_BatchNorm=False, 
                    input_shape_2d=(s, s, 6), output_shape_2d=(s, s, 1)).to(Device)

    # 训练要素：监督
    Loss_func = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=0)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=int(args.epochs*0.3), gamma=0.2)
    
    # 自监督训练组件
    Loss_self = nn.MSELoss()
    Optimizer_self = torch.optim.Adam(Net_model.parameters(), lr=0.0, betas=(0.7, 0.9), weight_decay=1e-4)
    
    # 初始化日志与可视化（epoch 0 前）
    log_loss = {'train': [], 'valid': [], 'train_self': []}
    start_time = time.time()
    
    Visual = MatplotlibVision(work_path, input_name=('1', '2', '3', '4', '5', '6'), field_name=('T',))


    # 训练循环（监督 + 自监督）
    for epoch in range(args.epochs):
        # 监督阶段
        Net_model.train()
        train_loss = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler)
        Net_model.eval()
        valid_loss = valid(valid_loader, Net_model, Device, Loss_func)
        log_loss['train'].append(train_loss)
        log_loss['valid'].append(valid_loss)
        if epoch < 300:
            lr_self = 0.0
        else:
            lr_self = float(args.self_lr_final) * (-epoch / max(args.epochs - 1, 1))
        for pg in Optimizer_self.param_groups:
            pg['lr'] = lr_self

        # 自监督阶段
        self_loss = train_selfsup(
            ss_loader,
            Net_model,
            Device,
            Loss_self,
            x_normalizer,
            y_normalizer,
            input_dim_mat,
            output_dim_mat,
            Optimizer_self,
            skip_backward=(lr_self == 0.0),
        )
        log_loss['train_self'].append(self_loss)

        elapsed = time.time() - start_time
        logger.info('epoch: {:6d}, lr_sup: {:.3e}, lr_self: {:.3e}, train_sup: {:.3e}, valid_sup: {:.3e}, train_self: {:.3e}, cost: {:.2f}'.
                    format(epoch, Optimizer.param_groups[0]['lr'], lr_self, train_loss, valid_loss, self_loss, elapsed))
        start_time = time.time()
            
        if epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['train'])), log_loss['train'], label='train_sup')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_sup')
            if len(log_loss['train_self']) > 0:
                Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_self'])), log_loss['train_self'], label='train_self')
            fig.suptitle('training loss (supervised + self-supervised)')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch % 50 == 0:
            train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device, down=args.down)
            valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device, down=args.down)

            train_true = y_normalizer.back(train_true)
            train_pred = y_normalizer.back(train_pred)
            valid_true = y_normalizer.back(valid_true)
            valid_pred = y_normalizer.back(valid_pred)
            
            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 
                        'optimizer': Optimizer.state_dict(), 'optimizer_self': Optimizer_self.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))
            np.save(os.path.join(work_path, 'loss_history.npy'), log_loss)

            for fig_id in range(5):
                fig, axs = plt.subplots(1, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], None)
                fig.savefig(os.path.join(work_path, f'train_solution_{str(fig_id)}_{str(epoch)}.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(1, 3, figsize=(18, 20), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], None)
                fig.savefig(os.path.join(work_path, f'valid_solution_{str(fig_id)}_{str(epoch)}.jpg'))
                plt.close(fig)
    
    # 训练完成
    logger.info('训练完成!')
    logger.info(f'最终 train_loss: {log_loss["train"][-1]:.6e}, valid_loss: {log_loss["valid"][-1]:.6e}, train_self_loss: {log_loss["train_self"][-1]:.6e}')