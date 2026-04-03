"""
BC组件化训练入口脚本 - 卫星热传导数据集

使用方法:
    python train_entry_bc.py --work_name test_bc \
        --train_component_nums 1,2 \
        --super_train_mode S0 --epochs 50 \
        --num_bc_components 1
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
from Demo.satellite_sup_2d.models_bc import supredictor_bc_component
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
    parser = argparse.ArgumentParser(description='BC组件化训练')
    
    # 数据参数
    parser.add_argument('--train_component_nums', type=str, default="1,2,3,4,5",
                        help='训练K桶列表')
    parser.add_argument('--valid_component_nums', type=str, 
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25",
                        help='验证K桶列表')
    parser.add_argument('--ntrain_perK', type=int, default=800)
    parser.add_argument('--nvalid_perK', type=int, default=40)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--base_path', type=str, default='/data/wqn/datasets/dataset_20251218_mc')
    
    # 叠加参数
    parser.add_argument('--super_train_mode', type=str, choices=['S0', 'S01', 'S012'], required=True)
    parser.add_argument('--super_nums_eval', type=str, default="0,1,2")
    parser.add_argument('--target_U_channels', type=int, default=16)
    parser.add_argument('--empty_channel_value', type=float, default=0.6)
    
    # BC组件化参数
    parser.add_argument('--num_bc_components', type=int, default=1,
                        help='BC组件数量: 1=单BC, M=多BC')
    parser.add_argument('--bc_dim', type=int, default=16,
                        help='BC组件输出维度')
    parser.add_argument('--bc_encoder_type', type=str, default='linear',
                        choices=['linear', 'cnn'],
                        help='BC编码器类型')
    parser.add_argument('--partition_strategy', type=str, default='sequential',
                        choices=['sequential', 'distribute'],
                        help='组件分组策略')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--scheduler_step', type=int, default=200)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    
    # 运行参数
    parser.add_argument('--work_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='runs_bc')
    parser.add_argument('--seed', type=int, default=8905)
    parser.add_argument('--gpu', type=str, default='0')
    
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list(s: str):
    return [int(x.strip()) for x in s.split(',')]


def main():
    args = parse_args()
    
    # 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    work_path = os.path.join(args.save_dir, args.work_name)
    os.makedirs(work_path, exist_ok=True)
    
    print("=" * 70)
    print(f"BC组件化训练: {args.work_name}")
    print("=" * 70)
    print(f"BC组件数: {args.num_bc_components}")
    print(f"BC维度: {args.bc_dim}")
    print(f"BC编码器: {args.bc_encoder_type}")
    
    # 保存配置
    with open(os.path.join(work_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 确定最大U通道数
    super_nums_train = {'S0': [0], 'S01': [0, 1], 'S012': [0, 1, 2]}[args.super_train_mode]
    max_super_num = max(super_nums_train)
    max_U_channels = args.target_U_channels * (2 ** max_super_num)
    
    train_component_nums = parse_list(args.train_component_nums)
    valid_component_nums = parse_list(args.valid_component_nums)

    print(f"训练模式: {args.super_train_mode}")
    print(f"最大U通道数: {max_U_channels}")
    print(f"ID验证K桶: {[k for k in valid_component_nums if k in set(train_component_nums)]}")
    print(f"OOD验证K桶: {[k for k in valid_component_nums if k not in set(train_component_nums)]}\n")

    # 加载数据
    print("加载数据集...")
    train_loader, valid_loaders, normalizers, meta = get_loaders_satellite_by_K(
        component_nums=valid_component_nums,
        train_component_nums=train_component_nums,
        ntrain_perK=args.ntrain_perK,
        nvalid_perK=args.nvalid_perK,
        target_U_channels=max_U_channels,
        empty_channel_value=args.empty_channel_value,
        batch_size=args.batch_size,
        base_path=args.base_path,
        downsample=args.downsample,
        split_by_K=True,
    )

    example_batch = next(iter(train_loader))
    example_G = example_batch[0][:1].cpu().numpy()
    parsed_G = parse_satellite_g_structure(example_G, meta)
    global_G_dim = parsed_G['global_G'].shape[-1]
    bc_sdf = parsed_G['bc_sdf']
    bc_input_channels = bc_sdf.shape[-1] // args.num_bc_components if bc_sdf is not None else meta['G_channels']

    if bc_sdf is not None and bc_sdf.shape[-1] % args.num_bc_components != 0:
        raise ValueError(
            f"bc_sdf通道数({bc_sdf.shape[-1]})无法被num_bc_components({args.num_bc_components})整除。"
        )
    if bc_sdf is None and meta['G_channels'] % args.num_bc_components != 0:
        raise ValueError(
            f"G通道数({meta['G_channels']})无法被num_bc_components({args.num_bc_components})整除，"
            "无法按多BC组件拆分。"
        )

    # 构建模型
    print("\n构建BC组件化模型...")
    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting_satellite()

    # 更新模型配置
    pred_model_dict['G_dim'] = global_G_dim
    pred_model_dict['U_dim'] = args.target_U_channels
    
    pred_model = DualHeadFourierTransformer(**pred_model_dict).to(device)
    super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(device)
    
    # 使用BC组件化模型包装
    model = supredictor_bc_component(
        pred_net=pred_model,
        super_net=super_model,
        channel_num=args.target_U_channels,
        G_channels=bc_input_channels,
        bc_dim=args.bc_dim,
        num_bc_components=args.num_bc_components,
        encoder_type=args.bc_encoder_type,
        partition_strategy=args.partition_strategy,
        g_meta=meta,
    ).to(device)
    
    print(f"  - Predictor: DualHeadFourierTransformer")
    print(f"  - Super model: FNO2d")
    print(f"  - Predictor G_dim: {global_G_dim}")
    print(f"  - BC组件数: {args.num_bc_components}")
    print(f"  - 每个BC组件输入通道: {bc_input_channels}")
    print(f"  - 数据总G通道: {meta['G_channels']}")
    print(f"  - BC维度: {args.bc_dim}")
    
    # 构建优化器
    print("\n构建优化器...")
    optimizers = {}
    schedulers = {}
    
    for s in super_nums_train:
        if s == 0:
            optimizers[s] = torch.optim.Adam(
                model.parameters(),
                lr=args.lr, betas=(0.7, 0.9), weight_decay=args.weight_decay
            )
        else:
            optimizers[s] = torch.optim.Adam(
                model.parameters(),
                lr=args.lr, betas=(0.7, 0.9), weight_decay=args.weight_decay
            )
        
        schedulers[s] = torch.optim.lr_scheduler.StepLR(
            optimizers[s], 
            step_size=args.scheduler_step, 
            gamma=args.scheduler_gamma
        )
        print(f"  - Optimizer[S{s}]: lr={optimizers[s].param_groups[0]['lr']:.2e}")
    
    # 初始化日志
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
    
    # 训练循环
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
            args.target_U_channels, device, loss_func,
            train_component_nums=train_component_nums,
        )

        # 记录
        for s in super_nums_train:
            log_loss['train'][s].append(train_metrics[s])
        for s in super_nums_eval:
            log_loss['valid'][s].append(valid_metrics['per_super'][s])

        logger.log_epoch(epoch, {
            'train': train_metrics,
            **valid_metrics
        })

        # 保存checkpoint
        avg_valid_id = np.mean([valid_metrics['per_super_id'][s] for s in super_nums_eval])
        if avg_valid_id < best_loss:
            best_loss = avg_valid_id
            torch.save(model.state_dict(), os.path.join(work_path, 'ckpt_best.pth'))
        torch.save(model.state_dict(), os.path.join(work_path, 'ckpt_last.pth'))

        # 可视化
        if epoch % 5 == 0:
            plot_loss_curves(
                log_loss,
                os.path.join(work_path, 'log_loss.svg'),
                super_nums_eval
            )

        # 打印进度
        epoch_time = time.time() - epoch_start
        train_str = ", ".join([f"S{s}={train_metrics[s]:.4e}" for s in super_nums_train])
        valid_all_str = ", ".join([f"S{s}={valid_metrics['per_super'][s]:.4e}" for s in super_nums_eval])
        valid_id_str = ", ".join([f"S{s}={valid_metrics['per_super_id'][s]:.4e}" for s in super_nums_eval])
        valid_ood_str = ", ".join([f"S{s}={valid_metrics['per_super_ood'][s]:.4e}" for s in super_nums_eval])

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train[{train_str}] | ValidAll[{valid_all_str}] | "
              f"ValidID[{valid_id_str}] | ValidOOD[{valid_ood_str}] | "
              f"BestID={best_loss:.4e} | Time={epoch_time:.1f}s")
    
    # 完成
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    logger.write_csv_summary()
    
    total_time = time.time() - start_time
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"最佳验证loss: {best_loss:.4e}")
    print(f"结果保存至: {work_path}")


if __name__ == "__main__":
    main()
