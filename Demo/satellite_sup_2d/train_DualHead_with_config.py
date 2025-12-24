import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import yaml
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


def load_dualhead_config(config_name='DualHead_GUT_2d'):
    """
    从 YAML 文件加载 DualHeadFourierTransformer 配置
    
    Args:
        config_name: 配置名称，对应 YAML 文件中的键
        
    Returns:
        config: 配置字典
    """
    config_path = os.path.join('data', 'configs', 'dualhead_transformer_config_sate.yml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please ensure the config file exists."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.full_load(f)
    
    if config_name not in all_configs:
        available = list(all_configs.keys())
        raise ValueError(
            f"Config '{config_name}' not found in {config_path}\n"
            f"Available configs: {available}"
        )
    
    config = all_configs[config_name]
    
    # 验证必需参数
    required = ['G_dim', 'U_dim', 'n_targets']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"Missing required parameters in config: {missing}")
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DualHeadFourierTransformer with YAML config'
    )
    parser.add_argument('--config', type=str, default='DualHead_GUT_2d',
                        help='Config name from dualhead_transformer_config_sate.yml')
    parser.add_argument('--h5', type=str,
                        default="/data/wqn/datasets/SDNO_test/15c_data_test.h5",
                        help='Path to h5 data file')
    parser.add_argument('--ntrain', type=int, default=4000, 
                        help='Number of training samples')
    parser.add_argument('--nvalid', type=int, default=1000, 
                        help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--scheduler_step', type=int, default=100, 
                        help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, 
                        help='Scheduler gamma')
    parser.add_argument('--work_name', type=str, default=None,
                        help='Work directory name (default: config name)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 加载配置
    print(f"Loading config: {args.config}")
    model_config = load_dualhead_config(args.config)
    
    # 工作路径
    work_name = args.work_name or f'DualHead_{args.config}'
    work_path = os.path.join('work_satellite', work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    Device = work.device
    
    # 记录配置
    Logger.write("=" * 60 + "\n")
    Logger.write(f"DualHeadFourierTransformer Training\n")
    Logger.write("=" * 60 + "\n")
    Logger.write(f"Config: {args.config}\n")
    Logger.write(f"Work path: {work_path}\n")
    Logger.write("\nModel configuration:\n")
    for key, value in sorted(model_config.items()):
        Logger.write(f"  {key:25s}: {value}\n")
    Logger.write("\nTraining configuration:\n")
    Logger.write(f"  ntrain:         {args.ntrain}\n")
    Logger.write(f"  nvalid:         {args.nvalid}\n")
    Logger.write(f"  batch_size:     {args.batch_size}\n")
    Logger.write(f"  epochs:         {args.epochs}\n")
    Logger.write(f"  lr:             {args.lr}\n")
    Logger.write(f"  scheduler_step: {args.scheduler_step}\n")
    Logger.write(f"  scheduler_gamma: {args.scheduler_gamma}\n")
    Logger.write("=" * 60 + "\n\n")
    
    # 数据加载
    G_dim = model_config['G_dim']
    U_dim = model_config['U_dim']
    
    Logger.write(f"Loading data...\n")
    Logger.write(f"  G_dim: {G_dim} (condition field channels)\n")
    Logger.write(f"  U_dim: {U_dim} (source field channels)\n")
    
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3, 4, 5],
        target_U_channels=U_dim,
        empty_channel_value=1.0,
        train_num=args.ntrain,
        valid_num=args.nvalid,
        batch_size=args.batch_size,
        shuffled=True,
        h5_path=args.h5,
    )
    
    Logger.write(f"  Training batches: {len(train_loader)}\n")
    Logger.write(f"  Validation batches: {len(valid_loader)}\n\n")
    
    # 创建模型
    Logger.write("Creating model...\n")
    Net_model = DualHeadFourierTransformer(**model_config).to(Device)
    
    # 统计参数
    total_params = sum(p.numel() for p in Net_model.parameters())
    trainable_params = sum(p.numel() for p in Net_model.parameters() if p.requires_grad)
    
    # 分组统计
    g_embed_params = sum(p.numel() for p in Net_model.g_downscaler.parameters())
    u_embed_params = sum(p.numel() for p in Net_model.u_downscaler.parameters())
    encoder_params = sum(p.numel() for p in Net_model.encoder_layers.parameters())
    regressor_params = sum(p.numel() for p in Net_model.regressor.parameters())
    
    Logger.write(f"  Total parameters:     {total_params:,}\n")
    Logger.write(f"  Trainable parameters: {trainable_params:,}\n")
    Logger.write(f"\n  Parameter breakdown:\n")
    Logger.write(f"    G embedding:  {g_embed_params:>8,} ({g_embed_params/total_params*100:>5.2f}%)\n")
    Logger.write(f"    U embedding:  {u_embed_params:>8,} ({u_embed_params/total_params*100:>5.2f}%)\n")
    Logger.write(f"    Encoders:     {encoder_params:>8,} ({encoder_params/total_params*100:>5.2f}%)\n")
    Logger.write(f"    Regressor:    {regressor_params:>8,} ({regressor_params/total_params*100:>5.2f}%)\n\n")
    
    # 损失与优化器
    Loss_func_train = nn.MSELoss()
    Loss_func_valid = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, 
                                 betas=(0.7, 0.9), weight_decay=1e-7)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, 
                                                step_size=args.scheduler_step, 
                                                gamma=args.scheduler_gamma)
    
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))
    
    # 训练循环
    Logger.write("Starting training...\n")
    Logger.write("=" * 60 + "\n")
    start_time = time.time()
    
    best_valid_loss = float('inf')
    
    for epoch in range(args.epochs):
        Net_model.train()
        train_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(Device), y_batch.to(Device)
            
            # 拆分输入: x_batch = [G, U] 拼接
            # G: 前 G_dim 个通道
            # U: 后 U_dim 个通道
            G = x_batch[:, :G_dim, :, :]      # [B, G_dim, H, W]
            U = x_batch[:, G_dim:G_dim+U_dim, :, :]  # [B, U_dim, H, W]
            
            # 前向传播
            pred = Net_model(G, U)
            
            # 计算损失
            loss = Loss_func_train(pred, y_batch)
            
            # 反向传播
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            Net_model.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    x_batch, y_batch = x_batch.to(Device), y_batch.to(Device)
                    G = x_batch[:, :G_dim, :, :]
                    U = x_batch[:, G_dim:G_dim+U_dim, :, :]
                    pred = Net_model(G, U)
                    loss = Loss_func_valid(pred, y_batch)
                    valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            
            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(Net_model.state_dict(), 
                          os.path.join(work_path, 'model_best.pth'))
                Logger.write(f"[Epoch {epoch:04d}] New best model saved!\n")
            
            Logger.write(f"[Epoch {epoch:04d}] Train: {train_loss:.6f} | "
                        f"Valid: {valid_loss:.6f} | "
                        f"Best: {best_valid_loss:.6f} | "
                        f"LR: {Scheduler.get_last_lr()[0]:.2e}\n")
        
        Scheduler.step()
    
    elapsed_time = time.time() - start_time
    
    Logger.write("=" * 60 + "\n")
    Logger.write(f"Training completed in {elapsed_time/60:.2f} minutes\n")
    Logger.write(f"Best validation loss: {best_valid_loss:.6f}\n")
    
    # 保存最终模型
    torch.save(Net_model.state_dict(), os.path.join(work_path, 'model_final.pth'))
    Logger.write(f"Final model saved to {work_path}/model_final.pth\n")
    Logger.write(f"Best model saved to {work_path}/model_best.pth\n")
    
    Logger.write("=" * 60 + "\n")
    Logger.write("Training finished!\n")

