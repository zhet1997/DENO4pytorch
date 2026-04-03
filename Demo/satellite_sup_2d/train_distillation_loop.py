"""
循环式叠加蒸馏主训练脚本

训练流程（每个round）：
1. 监督训练 + 一致性训练
2. 评估 m_C(n) 与 m_CS(n)
3. 判定teacher合格集合T与区间[A,B]
4. 生成伪标签shards
5. 蒸馏训练
6. 清理旧shards
7. 停止判定

使用方法:
    python train_distillation_loop.py --max_rounds 10 --supervised_epochs 50 --distill_epochs 20
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import copy

sys.path.append('/data/wqn/Code/DENO4pytorch')
sys.path.append('/data/wqn/Code/DENO4pytorch/Models')
sys.path.append('/data/wqn/Code/DENO4pytorch/Utilizes')

from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.DualHeadTransformer import DualHeadFourierTransformer
from Demo.satellite_sup_2d.ablation_satellite import (
    get_setting_satellite,
    get_loaders_satellite_multi_GUT,
)
from Demo.satellite_sup_2d.trains_satellite import (
    supredictor_list_windows, 
    train_supercondition, 
    valid_supercondition
)
from Demo.satellite_sup_2d.consistency_modules import PathSampler, ConsistencyTrainer
from Demo.satellite_sup_2d.evaluation_modules import (
    create_anchor_set,
    AnchorEvaluator,
    EligibilityGate,
    IntervalScheduler
)
from Demo.satellite_sup_2d.pseudo_label_modules import (
    UnlabeledSampler,
    TeacherManager,
    PseudoLabelBuilder,
    ShardManager
)
from Demo.satellite_sup_2d.distill_modules import (
    DistillTrainer,
    StopController,
    RoundLogger
)


def parse_args():
    parser = argparse.ArgumentParser(description='循环式叠加蒸馏训练')
    
    # 数据参数
    parser.add_argument('--ntrain', type=int, default=2000, help='训练样本数')
    parser.add_argument('--nvalid', type=int, default=500, help='验证样本数')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--component_nums', type=int, nargs='+', 
                        default=[1,2,3,4,5,6,7,8,9,10], help='元件数量列表')
    
    # Round循环参数
    parser.add_argument('--max_rounds', type=int, default=10, help='最大round数')
    parser.add_argument('--supervised_epochs', type=int, default=50, help='每round监督训练epochs')
    parser.add_argument('--consistency_epochs', type=int, default=10, help='每round一致性训练epochs')
    parser.add_argument('--distill_epochs', type=int, default=20, help='每round蒸馏训练epochs')
    
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_consistency', type=float, default=1e-4, help='一致性训练学习率')
    parser.add_argument('--lr_distill', type=float, default=5e-4, help='蒸馏训练学习率')
    
    # 一致性参数
    parser.add_argument('--K', type=int, default=8, help='多路径数量')
    parser.add_argument('--consistency_weight', type=float, default=1.0)
    
    # Teacher合格判定参数
    parser.add_argument('--tau_rmse', type=float, default=0.05, help='RMSE阈值')
    parser.add_argument('--delta_rmse', type=float, default=0.01, help='最小改进量')
    
    # 蒸馏参数
    parser.add_argument('--lambda_distill', type=float, default=0.5, help='伪标签损失权重')
    parser.add_argument('--samples_per_n', type=int, default=1000, help='每个n的伪标签样本数')
    
    # 停止条件
    parser.add_argument('--patience', type=int, default=3, help='n*停滞容忍轮数')
    
    # 工作路径
    parser.add_argument('--work_name', type=str, default='distillation_loop')
    
    # Anchor集参数
    parser.add_argument('--anchor_samples', type=int, default=50, help='每个n的anchor样本数')
    
    return parser.parse_args()


def train_supervised_phase(
    model, train_loader, valid_loader, optimizer, scheduler,
    device, loss_func, normalizers, epochs, in_dim, logger
):
    """监督训练阶段"""
    logger.write("\n" + "-"*60 + "\n")
    logger.write("监督训练阶段\n")
    logger.write("-"*60 + "\n")
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        # 确保所有参数可训练
        for param in model.parameters():
            param.requires_grad = True
        
        model.train()
        train_loss = train_supercondition(
            train_loader, model, device, loss_func,
            optimizer, scheduler,
            x_norm=normalizers['G'],
            super_num=0,
            channel_num=in_dim
        )
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        if valid_loader is not None:
            valid_loss = valid_supercondition(
                valid_loader, model, device, loss_func,
                x_norm=normalizers['G'],
                super_num=0,
                channel_num=in_dim
            )
            valid_losses.append(valid_loss)
        
        if (epoch + 1) % 10 == 0:
            msg = f"  Epoch {epoch+1}/{epochs}: train={train_loss:.6f}"
            if valid_loader:
                msg += f", valid={valid_loss:.6f}"
            logger.write(msg)
    
    final_train = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else np.mean(train_losses)
    final_valid = np.mean(valid_losses[-10:]) if len(valid_losses) >= 10 else (valid_losses[-1] if valid_losses else 0)
    
    logger.write(f"\n监督训练完成：final_train={final_train:.6f}, final_valid={final_valid:.6f}\n")
    
    return final_train, final_valid


def train_consistency_phase(
    model, consistency_trainer, train_loader, valid_loader,
    optimizer, epochs, logger
):
    """一致性训练阶段"""
    logger.write("\n" + "-"*60 + "\n")
    logger.write("一致性训练阶段（冻结C，更新S）\n")
    logger.write("-"*60 + "\n")
    
    train_losses = []
    train_stds = []
    valid_losses = []
    valid_stds = []
    
    for epoch in range(epochs):
        train_loss, train_std = consistency_trainer.train_one_epoch(
            train_loader, optimizer, log_interval=50
        )
        train_losses.append(train_loss)
        train_stds.append(train_std)
        
        if valid_loader is not None:
            valid_loss, valid_std = consistency_trainer.validate_one_epoch(valid_loader)
            valid_losses.append(valid_loss)
            valid_stds.append(valid_std)
        
        if (epoch + 1) % 5 == 0:
            msg = f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.6f}, std={train_std:.6f}"
            if valid_loader:
                msg += f", valid_loss={valid_loss:.6f}, valid_std={valid_std:.6f}"
            logger.write(msg)
    
    final_train_loss = np.mean(train_losses[-5:]) if len(train_losses) >= 5 else np.mean(train_losses)
    final_train_std = np.mean(train_stds[-5:]) if len(train_stds) >= 5 else np.mean(train_stds)
    
    logger.write(f"\n一致性训练完成：loss={final_train_loss:.6f}, std={final_train_std:.6f}\n")
    
    return final_train_loss, final_train_std


def train_distill_phase(
    model, distill_trainer, train_loader, valid_loader,
    pseudo_shard_path, optimizer, scheduler, normalizers, epochs, logger
):
    """蒸馏训练阶段"""
    logger.write("\n" + "-"*60 + "\n")
    logger.write("蒸馏训练阶段（混合真伪标签）\n")
    logger.write("-"*60 + "\n")
    
    total_losses = []
    real_losses = []
    pseudo_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        total_loss, real_loss, pseudo_loss = distill_trainer.train_one_epoch(
            train_loader, pseudo_shard_path, optimizer, scheduler,
            normalizers, log_interval=50
        )
        total_losses.append(total_loss)
        real_losses.append(real_loss)
        pseudo_losses.append(pseudo_loss)
        
        if valid_loader is not None:
            valid_loss = distill_trainer.validate_one_epoch(valid_loader, normalizers)
            valid_losses.append(valid_loss)
        
        if (epoch + 1) % 5 == 0:
            msg = f"  Epoch {epoch+1}/{epochs}: total={total_loss:.6f}, real={real_loss:.6f}, pseudo={pseudo_loss:.6f}"
            if valid_loader:
                msg += f", valid={valid_loss:.6f}"
            logger.write(msg)
    
    final_total = np.mean(total_losses[-5:]) if len(total_losses) >= 5 else np.mean(total_losses)
    final_real = np.mean(real_losses[-5:]) if len(real_losses) >= 5 else np.mean(real_losses)
    final_pseudo = np.mean(pseudo_losses[-5:]) if len(pseudo_losses) >= 5 else np.mean(pseudo_losses)
    
    logger.write(f"\n蒸馏训练完成：total={final_total:.6f}, real={final_real:.6f}, pseudo={final_pseudo:.6f}\n")
    
    return final_total, final_real, final_pseudo


def main():
    args = parse_args()
    
    # ==================== 初始化 ====================
    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting_satellite()
    
    # 工作路径
    work_path = os.path.join('work_satellite', args.work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'distillation_loop.log'))
    Device = work.device
    
    in_dim = basic_dict['in_dim']
    target_U_channels = 16
    
    Logger.write("\n" + "="*80 + "\n")
    Logger.write("循环式叠加蒸馏训练\n")
    Logger.write("="*80 + "\n")
    Logger.write(f"配置参数:\n")
    for key, value in vars(args).items():
        Logger.write(f"  {key}: {value}\n")
    Logger.write("="*80 + "\n")
    
    # 数据加载
    Logger.write("\n加载训练数据...\n")
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=args.component_nums,
        target_U_channels=target_U_channels,
        empty_channel_value=1.0,
        train_num=args.ntrain,
        valid_num=args.nvalid,
        batch_size=args.batch_size,
        shuffled=True
    )
    
    # 创建Anchor集
    Logger.write("\n创建Anchor评估集...\n")
    anchor_dict = create_anchor_set(
        component_nums=args.component_nums,
        samples_per_n=args.anchor_samples,
        target_U_channels=target_U_channels,
        seed=42
    )
    
    # 网络
    Logger.write("\n初始化网络...\n")
    pred_model = DualHeadFourierTransformer(**pred_model_dict).to(Device)
    super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(Device)
    Net_model = supredictor_list_windows(pred_model, super_model, channel_num=in_dim).to(Device)
    
    # 损失函数
    Loss_func = nn.MSELoss()
    
    # 初始化模块
    path_sampler = PathSampler(K=args.K, channel_num=in_dim)
    consistency_trainer = ConsistencyTrainer(Net_model, path_sampler, Device, K=args.K)
    
    evaluator = AnchorEvaluator(Net_model, anchor_dict, normalizers, Device)
    eligibility_gate = EligibilityGate(tau_rmse=args.tau_rmse, delta_rmse=args.delta_rmse)
    interval_scheduler = IntervalScheduler(initial_width=3, max_width=5)
    
    unlabeled_sampler = UnlabeledSampler(
        component_nums=args.component_nums,
        samples_per_n=args.samples_per_n
    )
    
    teacher_manager = TeacherManager(work_path)
    shard_manager = ShardManager(work_path, keep_last_n=2)
    
    stop_controller = StopController(patience=args.patience, max_n=max(args.component_nums))
    round_logger = RoundLogger(os.path.join(work_path, 'round_summary.log'))
    
    # ==================== 主循环 ====================
    Logger.write("\n" + "="*80 + "\n")
    Logger.write("开始Round循环训练\n")
    Logger.write("="*80 + "\n\n")
    
    for round_id in range(args.max_rounds):
        round_start_time = time.time()
        
        Logger.write("\n" + "#"*80 + "\n")
        Logger.write(f"Round {round_id}\n")
        Logger.write("#"*80 + "\n")
        
        round_metrics = {}
        
        # ==================== 1. 监督训练 ====================
        optimizer_sup = torch.optim.Adam(
            Net_model.parameters(),
            lr=args.lr,
            betas=(0.7, 0.9),
            weight_decay=1e-7
        )
        scheduler_sup = torch.optim.lr_scheduler.StepLR(
            optimizer_sup,
            step_size=train_dict['scheduler_step'],
            gamma=train_dict['scheduler_gamma']
        )
        
        train_sup_loss, valid_sup_loss = train_supervised_phase(
            Net_model, train_loader, valid_loader,
            optimizer_sup, scheduler_sup,
            Device, Loss_func, normalizers,
            args.supervised_epochs, in_dim, Logger
        )
        round_metrics['supervised_train_loss'] = train_sup_loss
        round_metrics['supervised_valid_loss'] = valid_sup_loss
        
        # ==================== 2. 一致性训练 ====================
        optimizer_cons = torch.optim.Adam(
            Net_model.super_net.parameters(),
            lr=args.lr_consistency,
            betas=(0.7, 0.9),
            weight_decay=1e-7
        )
        
        cons_loss, cons_std = train_consistency_phase(
            Net_model, consistency_trainer, train_loader, valid_loader,
            optimizer_cons, args.consistency_epochs, Logger
        )
        round_metrics['consistency_loss'] = cons_loss
        round_metrics['consistency_std'] = cons_std
        
        # ==================== 3. 评估m_C与m_CS ====================
        Logger.write("\n" + "-"*60 + "\n")
        Logger.write("评估Teacher合格性\n")
        Logger.write("-"*60 + "\n")
        
        m_C_dict = evaluator.evaluate_by_n(c_only=True)
        m_CS_dict = evaluator.evaluate_by_n(c_only=False)
        
        round_metrics['m_C'] = m_C_dict
        round_metrics['m_CS'] = m_CS_dict
        
        # ==================== 4. 判定T与[A,B] ====================
        T, n_star, gate_metrics = eligibility_gate.compute_eligible_set(m_C_dict, m_CS_dict)
        A, B = interval_scheduler.update_interval(T, n_star)
        
        round_metrics['T'] = T
        round_metrics['n_star'] = n_star
        round_metrics['interval_A'] = A
        round_metrics['interval_B'] = B
        
        # ==================== 5. 停止判定 ====================
        should_stop, stop_reason = stop_controller.should_stop(n_star, m_CS_dict, round_id)
        
        if should_stop:
            Logger.write(f"\n触发停止条件: {stop_reason}\n")
            round_logger.log_round(round_id, round_metrics)
            break
        
        # 如果T为空或区间无效，跳过蒸馏
        if not T or A is None or B is None:
            Logger.write("\n[警告] 可蒸馏集合为空或区间无效，跳过蒸馏阶段\n")
            round_logger.log_round(round_id, round_metrics)
            continue
        
        # ==================== 6. 生成伪标签 ====================
        Logger.write("\n" + "-"*60 + "\n")
        Logger.write("生成伪标签\n")
        Logger.write("-"*60 + "\n")
        
        # 保存teacher
        teacher_manager.save_teacher(Net_model, round_id)
        
        # 加载teacher（创建新实例避免干扰）
        teacher_model = copy.deepcopy(Net_model)
        teacher_model.eval()
        
        # 抽取无标签样本
        unlabeled_data = unlabeled_sampler.sample_for_interval(
            A, B,
            target_U_channels=target_U_channels
        )
        
        # 生成伪标签
        pseudo_builder = PseudoLabelBuilder(
            teacher_model, path_sampler, normalizers, Device, K=args.K
        )
        
        shard_path = shard_manager.get_shard_path(round_id)
        pseudo_stats = pseudo_builder.generate_shard(
            unlabeled_data, shard_path, round_id, batch_size=args.batch_size
        )
        
        round_metrics['pseudo_samples'] = pseudo_stats['total_samples']
        round_metrics['shard_size_mb'] = pseudo_stats['file_size_mb']
        
        # ==================== 7. 蒸馏训练 ====================
        optimizer_dist = torch.optim.Adam(
            Net_model.parameters(),
            lr=args.lr_distill,
            betas=(0.7, 0.9),
            weight_decay=1e-7
        )
        scheduler_dist = torch.optim.lr_scheduler.StepLR(
            optimizer_dist,
            step_size=train_dict['scheduler_step'],
            gamma=train_dict['scheduler_gamma']
        )
        
        distill_trainer = DistillTrainer(
            Net_model, Device, Loss_func, lambda_distill=args.lambda_distill
        )
        
        distill_total, distill_real, distill_pseudo = train_distill_phase(
            Net_model, distill_trainer, train_loader, valid_loader,
            shard_path, optimizer_dist, scheduler_dist,
            normalizers, args.distill_epochs, Logger
        )
        
        round_metrics['distill_total_loss'] = distill_total
        round_metrics['distill_real_loss'] = distill_real
        round_metrics['distill_pseudo_loss'] = distill_pseudo
        
        # ==================== 8. 清理旧shards ====================
        shard_manager.cleanup_old_shards(round_id)
        
        # ==================== 9. 保存round信息 ====================
        round_metrics['round_time'] = time.time() - round_start_time
        round_logger.log_round(round_id, round_metrics)
        
        # 保存模型
        torch.save({
            'round': round_id,
            'model_state': Net_model.state_dict(),
            'n_star': n_star,
            'T': T,
            'interval': (A, B),
            'm_C': m_C_dict,
            'm_CS': m_CS_dict,
        }, os.path.join(work_path, f'model_round_{round_id}.pth'))
        
        Logger.write(f"\nRound {round_id} 完成，耗时: {round_metrics['round_time']:.2f}s\n")
    
    # ==================== 训练完成 ====================
    Logger.write("\n" + "="*80 + "\n")
    Logger.write("训练完成！\n")
    Logger.write("="*80 + "\n")
    
    # 保存最终模型
    torch.save(Net_model, work.fpth)
    
    # 保存训练摘要
    round_logger.save_summary(os.path.join(work_path, 'training_summary.json'))
    
    Logger.write(f"\n最终状态:\n")
    Logger.write(f"  best_n_star: {stop_controller.best_n_star}\n")
    Logger.write(f"  n_star_history: {stop_controller.n_star_history}\n")
    Logger.write(f"  总轮数: {len(stop_controller.n_star_history)}\n")
    
    print("\n训练完成！详细日志请查看: " + os.path.join(work_path, 'distillation_loop.log'))


if __name__ == "__main__":
    main()


