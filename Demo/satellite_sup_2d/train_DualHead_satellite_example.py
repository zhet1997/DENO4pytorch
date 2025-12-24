import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

sys.path.append('/data/wqn/DENO4pytorch')
sys.path.append('/data/wqn/DENO4pytorch/Models')
sys.path.append('/data/wqn/DENO4pytorch/Utilizes')

from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
from Tools.train_model.model_whole_life import WorkPrj
from transformer.DualHeadTransformer import DualHeadFourierTransformer

from Demo.satellite_sup_2d.ablation_satellite import get_loaders_satellite_multi_GUT
from Demo.satellite_sup_2d.trains_satellite import train_base_GUT as train_base, valid_base_GUT as valid_base


def get_dualhead_config():
    """
    DualHeadFourierTransformer 的配置
    
    关键区别:
        - 使用 G_dim 和 U_dim 而不是 node_feats
        - G 和 U 在模型内部分别处理，而不是预先拼接
    """
    config = dict(
        # 必需参数
        G_dim=4,           # G 的通道数 (4个通道: 可能是 x, y, scale_x, scale_y)
        U_dim=1,           # U 的通道数 (1个通道: 源项场)
        n_targets=1,       # 输出通道数 (1个通道: 温度场)
        
        # 网络结构
        n_hidden=96,
        num_encoder_layers=4,
        n_head=4,
        dim_feedforward=384,  # 2 * n_hidden
        
        # 注意力类型
        attention_type='fourier',  # 'fourier', 'galerkin', 'linear', 'softmax'
        
        # 解码器
        decoder_type='pointwise',  # 'pointwise' or 'ifft'
        num_regressor_layers=2,
        
        # 空间维度
        spacial_dim=2,      # 2D
        pos_dim=0,          # 不需要，因为 pos 已在 G 中
        
        # 正则化
        dropout=0.05,
        encoder_dropout=0.05,
        decoder_dropout=0.05,
        ffn_dropout=0.05,
        
        # 激活函数
        activation_type='silu',
        regressor_activation='silu',
        
        # 其他
        xavier_init=1e-2,
        diagonal_weight=1e-2,
        symmetric_init=False,
        norm_eps=1e-5,
        spacial_fc=False,
        
        # 调试
        return_latent=False,
        debug=False,
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train DualHeadFourierTransformer on satellite data')
    parser.add_argument('--h5', type=str,
                        default="/data/wqn/datasets/SDNO_test/15c_data_test.h5",
                        help='Path to h5 data file')
    parser.add_argument('--ntrain', type=int, default=4000, help='Number of training samples')
    parser.add_argument('--nvalid', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--work_name', type=str, default='DualHead_satellite_example',
                        help='Work directory name')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 配置
    model_config = get_dualhead_config()
    
    # 工作路径
    work_path = os.path.join('work_satellite', args.work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    Device = work.device
    
    Logger.write(f"DualHeadFourierTransformer Configuration:\n")
    for key, value in model_config.items():
        Logger.write(f"  {key}: {value}\n")
    
    # 数据加载
    # 注意: 数据加载器应该返回分离的 G 和 U
    # 如果你的数据加载器返回拼接的数据，需要在训练循环中手动拆分
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3, 4, 5],
        target_U_channels=1,  # U_dim
        empty_channel_value=1.0,
        train_num=args.ntrain,
        valid_num=args.nvalid,
        batch_size=args.batch_size,
        shuffled=True
    )
    
    Logger.write(f"\nData info:\n")
    Logger.write(f"  Training samples: {args.ntrain}\n")
    Logger.write(f"  Validation samples: {args.nvalid}\n")
    Logger.write(f"  Batch size: {args.batch_size}\n")
    
    # 创建模型
    Net_model = DualHeadFourierTransformer(**model_config).to(Device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in Net_model.parameters())
    trainable_params = sum(p.numel() for p in Net_model.parameters() if p.requires_grad)
    Logger.write(f"\nModel info:\n")
    Logger.write(f"  Total parameters: {total_params:,}\n")
    Logger.write(f"  Trainable parameters: {trainable_params:,}\n")
    
    # 损失与优化器
    Loss_func_train = nn.MSELoss()
    Loss_func_valid = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, 
                                 betas=(0.7, 0.9), weight_decay=1e-7)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=100, gamma=0.5)
    
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))
    
    # 训练循环
    Logger.write(f"\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        Net_model.train()
        train_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(Device), y_batch.to(Device)
            
            # 拆分输入: x_batch 包含 [G, U] 拼接
            # 假设 G 有 4 个通道，U 有 1 个通道
            G = x_batch[:, :4, :, :]  # [B, 4, H, W]
            U = x_batch[:, 4:5, :, :]  # [B, 1, H, W]
            
            # 前向传播
            pred = Net_model(G, U)  # [B, 1, H, W]
            
            # 计算损失
            loss = Loss_func_train(pred, y_batch)
            
            # 反向传播
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        if epoch % 10 == 0:
            Net_model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    x_batch, y_batch = x_batch.to(Device), y_batch.to(Device)
                    G = x_batch[:, :4, :, :]
                    U = x_batch[:, 4:5, :, :]
                    pred = Net_model(G, U)
                    loss = Loss_func_valid(pred, y_batch)
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            
            Logger.write(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | "
                        f"Valid Loss: {valid_loss:.6f} | LR: {Scheduler.get_last_lr()[0]:.2e}\n")
        
        Scheduler.step()
    
    elapsed_time = time.time() - start_time
    Logger.write(f"\nTraining completed in {elapsed_time/60:.2f} minutes\n")
    
    # 保存模型
    torch.save(Net_model.state_dict(), os.path.join(work_path, 'model_final.pth'))
    Logger.write(f"Model saved to {work_path}/model_final.pth\n")

