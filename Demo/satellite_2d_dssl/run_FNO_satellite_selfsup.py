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

from fno.FNOs import FNO2d
from Utilizes.process_data import DataNormer
from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
from Demo.satellite_2d_dssl.dataset_selfsup import (
    load_selfsup_data,
    dimension_scaling_Tensor,
)
from Demo.satellite_2d_base.utils import load_yaml_config
from Utilizes.visual_data import MatplotlibVision



def feature_transform(x):
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
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
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
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
                  train_loss: float,
                  loss_gap: float = 0.1) -> float:
    """
    自监督训练阶段：基于线性物理相似性
    - 对 inputs 随机选取一组 alpha 做相似映射得到 x_far
    - 分别预测 y_anc, y_far
    - 将 y_far 反映射到锚点尺度，与 y_anc 计算 MSE 损失
    - 当 self_loss/train_loss < loss_gap 时跳过 backward（一致性已足够好）
    
    参数:
        train_loss: 当前 epoch 的监督训练损失
        loss_gap: 损失比例阈值，低于此值时跳过 backward（默认 0.1）
    """
    netmodel.train()
    epoch_loss = 0.0
    skipped_batches = 0
    for batch, (xx, alpha_all) in enumerate(dataloader_ss):
        # xx: (B,H,W,C) 已归一化的tensor, alpha_all: (B,K,4) tensor
        xx = xx.to(device=device, dtype=torch.float32) #这个是归一化之后的数据
        alpha_all = alpha_all.to(device=device, dtype=torch.float32)

        # 反标准化到物理空间
        x_den_anc = x_normalizer.back(xx)

        # 为每个样本随机选择一组 alpha: (B,4)
        alpha = _pick_random_alpha(alpha_all)

        # 输入相似映射：在物理空间执行，再标准化用于网络输入
        x_den_far = dimension_scaling_Tensor(x_den_anc, bc_dim_mat=input_dim_mat, bc_dim_coef=alpha)
        x_norm_anc = xx
        x_norm_far = x_normalizer.norm(x_den_far)

        # 前向传播
        gd_anc = feature_transform(x_norm_anc)
        gd_far = gd_anc  # 使用相同网格
        y_norm_anc = netmodel(x_norm_anc, gd_anc)
        y_norm_far = netmodel(x_norm_far, gd_far)

        # 回到物理空间，对齐尺度，再标准化
        y_den_far = y_normalizer.back(y_norm_far)
        y_den_far_back = dimension_scaling_Tensor(y_den_far, bc_dim_mat=output_dim_mat, bc_dim_coef=-alpha)
        y_norm_far_back = y_normalizer.norm(y_den_far_back)

        # 自监督损失
        loss = lossfunc(y_norm_far_back, y_norm_anc)
        
        # 计算损失比例，决定是否执行 backward
        loss_ratio = float(loss.item()) / max(train_loss, 1e-10)
        if loss_ratio >= loss_gap:
            optimizer_ss.zero_grad()
            loss.backward()
            optimizer_ss.step()
        else:
            skipped_batches += 1

        epoch_loss += float(loss.item())

    avg_loss = epoch_loss / (batch + 1)
    if skipped_batches > 0:
        print(f"  [自监督] 跳过 {skipped_batches}/{batch+1} batches (loss_ratio < {loss_gap})")
    return avg_loss


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        gd = feature_transform(xx)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FNO 卫星热数据训练脚本（监督 + 自监督：线性物理相似性）')
    parser.add_argument('--data_path', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5')
    parser.add_argument('--ntrain', type=int, default=5000)
    parser.add_argument('--nvalid', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=7)
    parser.add_argument('--modes_x', type=int, default=10)
    parser.add_argument('--modes_y', type=int, default=10)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--work_dir', type=str, default=os.path.join('work_satellite'))
    # 自监督参数
    parser.add_argument('--selfsup_dir', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c_sim1/')
    parser.add_argument('--self_batch_size', type=int, default=32)
    parser.add_argument('--self_lr_final', type=float, default=1e-4)
    parser.add_argument('--self_sample_limit', type=int, default=None)
    args = parser.parse_args()

    net_name = 'FNO_DSSL'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_path = os.path.join(args.work_dir, f'{net_name}_n{args.ntrain}_{timestamp}')
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

    # 设备
    if torch.cuda.is_available():
        Device = torch.device(f'cuda:{args.cuda_index}')
        logger.info(f'使用设备: {Device}')
    else:
        Device = torch.device('cpu')

    # 加载数据（inputs: N,256,256,6; outputs: N,256,256,1）
    inputs, outputs = load_satellite_data(args.data_path)
    N = inputs.shape[0]

    ntrain = args.ntrain
    nvalid = args.nvalid
    assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'
    
    down = 4

    train_x = torch.tensor(inputs[:ntrain, ::down, ::down, :], dtype=torch.float32)
    train_y = torch.tensor(outputs[:ntrain, ::down, ::down, :], dtype=torch.float32)
    valid_x = torch.tensor(inputs[N - nvalid:, ::down, ::down, :], dtype=torch.float32)
    valid_y = torch.tensor(outputs[N - nvalid:, ::down, ::down, :], dtype=torch.float32)

    # 归一化（主程序）
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    
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

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    modes = (args.modes_x, args.modes_y)
    Net_model = FNO2d(in_dim=6, out_dim=1, modes=modes, width=args.width, depth=args.depth, steps=args.steps,
                      padding=args.padding, activation='gelu').to(Device)

    # 训练要素：监督
    Loss_func = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=int(args.epochs*0.8), gamma=0.1)

    # 自监督数据加载（不含 temperature）
    ss_inputs_np, ss_alphas_np = load_selfsup_data(args.selfsup_dir, sample_limit=args.self_sample_limit)
    logger.info(f"自监督数据: ss_inputs shape={ss_inputs_np.shape}, ss_alphas shape={ss_alphas_np.shape}")
    # 与监督相同下采样
    ss_inputs_np = ss_inputs_np[:, ::down, ::down, :].astype(np.float32)
    ss_alphas_np = ss_alphas_np.astype(np.float32)
    
    # 将自监督数据转换为tensor并进行归一化
    ss_inputs_tensor = torch.tensor(ss_inputs_np, dtype=torch.float32)
    ss_alphas_tensor = torch.tensor(ss_alphas_np, dtype=torch.float32)
    
    # 使用监督训练集的归一化器对自监督输入进行归一化
    ss_inputs_normalized = x_normalizer.norm(ss_inputs_tensor)
    
    # 创建自监督数据集（使用归一化后的tensor）
    ss_dataset = TensorDataset(ss_inputs_normalized, ss_alphas_tensor)
    ss_loader = DataLoader(ss_dataset, batch_size=args.self_batch_size, shuffle=True, drop_last=False)

    # 量纲矩阵：输入从新版本YAML，输出沿用 dataset_satellite 定义（temperature）
    yaml_path = os.path.join(CURRENT_DIR, 'augmentation_satellite.yml')
    input_dim_mat_np = build_input_dim_matrix_from_yaml_v2(yaml_path)  # (4,C)
    input_dim_mat = torch.tensor(input_dim_mat_np, dtype=torch.float32, device=Device)
    logger.info(f"输入量纲矩阵形状: {input_dim_mat.shape}")
    
    output_dim_mat = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32, device=Device)
    logger.info(f"输出量纲矩阵形状: {output_dim_mat.shape}")
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

        # 设置自监督学习率：线性从 0 → self_lr_final（按 epoch 比例）
        lr_self = float(args.self_lr_final) * (epoch / max(args.epochs - 1, 1))
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
            train_loss=train_loss,
            loss_gap=0.1,
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
            train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
            
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