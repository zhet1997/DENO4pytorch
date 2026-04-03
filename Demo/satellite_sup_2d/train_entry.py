"""
训练入口脚本 - 卫星热传导数据集叠加消融实验

使用方法:
    python train_entry.py --work_name test_kmax10_S01 \
        --train_component_nums 1,2,3,4,5,6,7,8,9,10 \
        --super_train_mode S01 --epochs 200
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append('/data/wqn/Code/DENO4pytorch')
sys.path.append('/data/wqn/Code/DENO4pytorch/Models')
sys.path.append('/data/wqn/Code/DENO4pytorch/Utilizes')

from Utilizes.visual_data import MatplotlibVision
from fno.FNOs import FNO2d
from transformer.DualHeadTransformer import DualHeadFourierTransformer
from Demo.satellite_sup_2d.trains_satellite import supredictor_list_windows
from Demo.satellite_sup_2d.ablation_satellite import get_setting_satellite
from Demo.satellite_sup_2d.data_loader_satellite import (
    get_loaders_satellite_by_K,
    parse_satellite_g_structure,
)
from Demo.satellite_sup_2d.trainer_satellite import (
    train_one_epoch, validate_all, inference_sample,
    plot_loss_curves, plot_field_comparison, MetricsLogger
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='卫星数据叠加消融实验')
    
    # 数据参数
    parser.add_argument('--train_component_nums', type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help='训练用的K桶列表，逗号分隔')
    parser.add_argument('--valid_component_nums', type=str, 
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25",
                        help='验证用的K桶列表，逗号分隔')
    parser.add_argument('--ntrain_perK', type=int, default=800,
                        help='每个K桶的训练样本数')
    parser.add_argument('--nvalid_perK', type=int, default=40,
                        help='每个K桶的验证样本数')
    parser.add_argument('--downsample', type=int, default=2,
                        help='下采样因子: 1=256, 2=128, 4=64')
    parser.add_argument('--base_path', type=str, default='/data/wqn/datasets/dataset_20251218_mc',
                        help='数据集基础路径（兼容旧版与_mc新版命名）')
    
    # 叠加参数
    parser.add_argument('--super_train_mode', type=str, choices=['S0', 'S01', 'S012'], required=True,
                        help='训练模式: S0/S01/S012')
    parser.add_argument('--super_nums_eval', type=str, default="0,1,2",
                        help='评估的super_num列表，逗号分隔')
    parser.add_argument('--target_U_channels', type=int, default=16,
                        help='基础通道数（channel_num）')
    parser.add_argument('--empty_channel_value', type=float, default=0.6,
                        help='空通道填充值')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='权重衰减')
    parser.add_argument('--scheduler_step', type=int, default=200,
                        help='学习率衰减步长')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='学习率衰减系数')
    
    # 运行参数
    parser.add_argument('--work_name', type=str, required=True,
                        help='实验名称')
    parser.add_argument('--save_dir', type=str, default='runs',
                        help='保存目录')
    parser.add_argument('--seed', type=int, default=8905,
                        help='随机种子')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU编号')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list(s: str):
    """解析逗号分隔的字符串为整数列表"""
    return [int(x.strip()) for x in s.split(',')]


def main():
    args = parse_args()
    
    # 0. 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    work_path = os.path.join(args.save_dir, args.work_name)
    os.makedirs(work_path, exist_ok=True)
    
    print("=" * 70)
    print(f"实验: {args.work_name}")
    print("=" * 70)
    
    # 保存配置
    with open(os.path.join(work_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 1. 确定最大U通道数
    super_nums_train = {'S0': [0], 'S01': [0, 1], 'S012': [0, 1, 2]}[args.super_train_mode]
    max_super_num = max(super_nums_train)
    max_U_channels = args.target_U_channels * (2 ** max_super_num)
    
    print(f"训练模式: {args.super_train_mode}")
    print(f"训练super_nums: {super_nums_train}")
    print(f"最大U通道数: {max_U_channels}")
    print(f"设备: {device}\n")
    
    # 2. 加载数据
    print("加载数据集...")
    train_loader, valid_loaders, normalizers, meta = get_loaders_satellite_by_K(
        component_nums=parse_list(args.valid_component_nums),
        train_component_nums=parse_list(args.train_component_nums),
        ntrain_perK=args.ntrain_perK,
        nvalid_perK=args.nvalid_perK,
        target_U_channels=max_U_channels,
        empty_channel_value=args.empty_channel_value,
        batch_size=args.batch_size,
        base_path=args.base_path,
        downsample=args.downsample,
        split_by_K=True,
    )
    
    # 3. 构建模型
    print("\n构建模型...")
    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting_satellite()

    example_batch = next(iter(train_loader))
    example_G = example_batch[0][:1].cpu().numpy()
    parsed_G = parse_satellite_g_structure(example_G, meta)
    global_G_dim = parsed_G['global_G'].shape[-1]

    # 更新模型配置中的输入维度
    pred_model_dict['G_dim'] = global_G_dim
    pred_model_dict['U_dim'] = args.target_U_channels
    
    pred_model = DualHeadFourierTransformer(**pred_model_dict).to(device)
    super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(device)
    model = supredictor_list_windows(
        pred_model, super_model,
        channel_num=args.target_U_channels,
        g_meta=meta,
    ).to(device)
    
    print(f"  - Predictor: DualHeadFourierTransformer")
    print(f"  - Super model: FNO2d")
    print(f"  - Predictor G_dim: {global_G_dim}")
    print(f"  - Channel_num: {args.target_U_channels}")
    
    # 4. 构建优化器
    print("\n构建优化器...")
    optimizers = {}
    schedulers = {}
    
    for s in super_nums_train:
        if s == 0:
            optimizers[s] = torch.optim.Adam(
                model.pred_net.parameters(),
                lr=args.lr, betas=(0.7, 0.9), weight_decay=args.weight_decay
            )
        else:
            optimizers[s] = torch.optim.Adam([
                {'params': model.pred_net.parameters(), 'lr': args.lr * (0.3 ** s)},
                {'params': model.super_net.parameters()}
            ], lr=args.lr, betas=(0.7, 0.9), weight_decay=args.weight_decay)
        
        schedulers[s] = torch.optim.lr_scheduler.StepLR(
            optimizers[s], 
            step_size=args.scheduler_step, 
            gamma=args.scheduler_gamma
        )
        print(f"  - Optimizer[S{s}]: lr={optimizers[s].param_groups[0]['lr']:.2e}")
    
    # 5. 初始化日志
    logger = MetricsLogger(work_path)
    visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))
    loss_func = nn.MSELoss()
    
    super_nums_eval = parse_list(args.super_nums_eval)
    if any(s > max_super_num for s in super_nums_eval):
        print(f"警告: super_nums_eval={super_nums_eval} 包含超过训练上限的值，已裁剪为不超过 {max_super_num}")
        super_nums_eval = [s for s in super_nums_eval if s <= max_super_num]
    log_loss = {
        'train': {s: [] for s in super_nums_train},
        'valid': {s: [] for s in super_nums_eval},
    }
    
    # 6. 训练循环
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_one_epoch(
            train_loader, model, optimizers, schedulers,
            super_nums_train, args.target_U_channels, device, loss_func
        )
        
        # 验证
        valid_metrics = validate_all(
            valid_loaders, model, super_nums_eval,
            args.target_U_channels, device, loss_func
        )
        
        # 记录loss
        for s in super_nums_train:
            log_loss['train'][s].append(train_metrics[s])
        for s in super_nums_eval:
            log_loss['valid'][s].append(valid_metrics['per_super'][s])
        
        logger.log_epoch(epoch, {
            'train': train_metrics,
            **valid_metrics
        })
        
        # 保存checkpoint
        avg_valid = np.mean([valid_metrics['per_super'][s] for s in super_nums_eval])
        if avg_valid < best_loss:
            best_loss = avg_valid
            torch.save(model.state_dict(), os.path.join(work_path, 'ckpt_best.pth'))
        torch.save(model.state_dict(), os.path.join(work_path, 'ckpt_last.pth'))
        
        # 可视化：收敛曲线（每5epoch）
        if epoch % 5 == 0:
            plot_loss_curves(
                log_loss, 
                os.path.join(work_path, 'log_loss.svg'), 
                super_nums_eval
            )
        
        # 可视化：随机采样（每50epoch）
        if epoch % 200 == 0:
            for s in super_nums_eval:
                try:
                    # Train samples
                    coords_tr, true_tr, pred_tr = inference_sample(
                        train_loader, model, normalizers, s,
                        args.target_U_channels, device, num_samples=5
                    )
                    for i in range(min(5, len(coords_tr))):
                        plot_field_comparison(
                            coords_tr[i], true_tr[i], pred_tr[i],
                            os.path.join(work_path, f'train_s{s}_sample{i}_ep{epoch}.jpg'),
                            visual
                        )
                    
                    # Valid samples (从K=10桶采样，如果存在)
                    if 10 in valid_loaders:
                        coords_vd, true_vd, pred_vd = inference_sample(
                            valid_loaders[10], model, normalizers, s,
                            args.target_U_channels, device, num_samples=5
                        )
                        for i in range(min(5, len(coords_vd))):
                            plot_field_comparison(
                                coords_vd[i], true_vd[i], pred_vd[i],
                                os.path.join(work_path, f'valid_s{s}_sample{i}_ep{epoch}.jpg'),
                                visual
                            )
                except Exception as e:
                    print(f"  警告: 可视化失败 (S{s}, epoch {epoch}): {e}")
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        train_str = ", ".join([f"S{s}={train_metrics[s]:.4e}" for s in super_nums_train])
        valid_str = ", ".join([f"S{s}={valid_metrics['per_super'][s]:.4e}" for s in super_nums_eval])
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train[{train_str}] | Valid[{valid_str}] | "
              f"Best={best_loss:.4e} | Time={epoch_time:.1f}s")
        
        # 每10个epoch详细打印分桶验证loss（用于诊断NaN问题）
        if epoch % 10 == 0:
            print(f"  Validation details (per K bucket):")
            for s in super_nums_eval:
                K_losses = valid_metrics['per_K'][s]
                K_str = ", ".join([f"K{k}={v:.3f}" if not np.isnan(v) else f"K{k}=NaN" 
                                   for k, v in sorted(K_losses.items())])
                print(f"    S{s}: {K_str}")
    
    # 7. 最终导出
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    logger.write_csv_summary()
    
    total_time = time.time() - start_time
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"最佳验证loss: {best_loss:.4e}")
    print(f"结果保存至: {work_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

