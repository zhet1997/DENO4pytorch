"""
蒸馏训练模块

模块功能：
1. DistillTrainer: 混合真标签与伪标签训练学生模型
2. StopController: 外循环停止控制器

设计思路：
- 混合损失: L = (1-λ)*L_real + λ*L_pseudo
- 每个batch混合真伪标签
- 停止条件: n*连续停滞patience轮
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DistillTrainer:
    """
    混合蒸馏训练器
    
    功能：
    - 混合真标签数据与伪标签数据训练学生模型
    - 损失权重可调：L = (1-λ)*L_real + λ*L_pseudo
    - 支持动态混合比例调度
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_func: nn.Module,
        lambda_distill: float = 0.5
    ):
        """
        Args:
            model: 学生模型（C+S）
            device: 训练设备
            loss_func: 损失函数
            lambda_distill: 伪标签损失权重 (0~1)
        """
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.lambda_distill = lambda_distill
    
    def train_one_epoch(
        self,
        real_loader,
        pseudo_shard_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        normalizers: Dict[str, any],
        log_interval: int = 20
    ) -> Tuple[float, float, float]:
        """
        训练一个epoch（混合真伪标签）
        
        Args:
            real_loader: 真标签数据加载器 (G, U, T)
            pseudo_shard_path: 伪标签HDF5 shard路径
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            normalizers: {'G': ..., 'U': ..., 'T': ...}
            log_interval: 日志打印间隔
        
        Returns:
            avg_total_loss: 平均总损失
            avg_real_loss: 平均真标签损失
            avg_pseudo_loss: 平均伪标签损失
        """
        self.model.train()
        
        # 加载伪标签shard
        with h5py.File(pseudo_shard_path, 'r') as f:
            G_pseudo = f['G'][:]
            U_pseudo = f['U'][:]
            T_pseudo = f['T_pseudo'][:]
        
        # 归一化伪标签
        # 注意：T_pseudo已经是归一化后的输出，无需再次归一化
        G_pseudo_norm = normalizers['G'].norm(G_pseudo)
        U_pseudo_norm = normalizers['U'].norm(U_pseudo)
        T_pseudo_norm = T_pseudo  # 已归一化
        
        # 创建伪标签loader
        pseudo_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(G_pseudo_norm, dtype=torch.float),
            torch.as_tensor(U_pseudo_norm, dtype=torch.float),
            torch.as_tensor(T_pseudo_norm, dtype=torch.float)
        )
        
        batch_size = real_loader.batch_size
        pseudo_loader = torch.utils.data.DataLoader(
            pseudo_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # 训练循环（混合真伪标签）
        total_loss_sum = 0.0
        real_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        num_batches = 0
        
        # 创建交替迭代器
        real_iter = iter(real_loader)
        pseudo_iter = iter(pseudo_loader)
        
        # 以较短的loader为准
        max_batches = min(len(real_loader), len(pseudo_loader))
        
        for batch_idx in range(max_batches):
            try:
                G_r, U_r, T_r = next(real_iter)
                G_p, U_p, T_p = next(pseudo_iter)
            except StopIteration:
                break
            
            # 真标签损失
            G_r = G_r.to(self.device)
            U_r = U_r.to(self.device)
            T_r = T_r.to(self.device)
            pred_r = self.model(G_r, U_r)
            loss_r = self.loss_func(pred_r, T_r)
            
            # 伪标签损失
            G_p = G_p.to(self.device)
            U_p = U_p.to(self.device)
            T_p = T_p.to(self.device)
            pred_p = self.model(G_p, U_p)
            loss_p = self.loss_func(pred_p, T_p)
            
            # 混合损失
            loss = (1 - self.lambda_distill) * loss_r + self.lambda_distill * loss_p
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss_sum += loss.item()
            real_loss_sum += loss_r.item()
            pseudo_loss_sum += loss_p.item()
            num_batches += 1
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch [{batch_idx+1}/{max_batches}] "
                      f"Total: {loss.item():.6f}, "
                      f"Real: {loss_r.item():.6f}, "
                      f"Pseudo: {loss_p.item():.6f}")
        
        if scheduler is not None:
            scheduler.step()
        
        avg_total_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_real_loss = real_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_pseudo_loss = pseudo_loss_sum / num_batches if num_batches > 0 else 0.0
        
        return avg_total_loss, avg_real_loss, avg_pseudo_loss
    
    def validate_one_epoch(
        self,
        valid_loader,
        normalizers: Dict[str, any]
    ) -> float:
        """
        验证一个epoch（只用真标签）
        
        Args:
            valid_loader: 验证数据加载器
            normalizers: 归一化器
        
        Returns:
            avg_loss: 平均验证损失
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for G, U, T in valid_loader:
                G = G.to(self.device)
                U = U.to(self.device)
                T = T.to(self.device)
                
                pred = self.model(G, U)
                loss = self.loss_func(pred, T)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


class StopController:
    """
    外循环停止控制器
    
    功能：
    - 监控n*推进情况
    - 判断是否应该停止训练
    
    停止条件：
    1. n*连续patience轮未推进
    2. n*达到component_nums上限
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_n_star_gain: int = 1,
        max_n: Optional[int] = None
    ):
        """
        Args:
            patience: n*停滞容忍轮数
            min_n_star_gain: 最小n*增长量（判定推进）
            max_n: component_num上限（如果达到则停止）
        """
        self.patience = patience
        self.min_n_star_gain = min_n_star_gain
        self.max_n = max_n
        self.stagnant_count = 0
        self.best_n_star = 0
        self.n_star_history = []
    
    def should_stop(
        self,
        n_star: int,
        m_CS_dict: Optional[Dict[int, float]] = None,
        current_round: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        判断是否应该停止训练
        
        Args:
            n_star: 当前可靠前沿
            m_CS_dict: 当前RMSE字典（可选，用于额外诊断）
            current_round: 当前轮次（可选，用于日志）
        
        Returns:
            should_stop: 是否停止
            reason: 停止原因
        """
        self.n_star_history.append(n_star)
        
        # 条件1：n*推进检测
        if n_star > self.best_n_star:
            # n*推进
            gain = n_star - self.best_n_star
            self.best_n_star = n_star
            self.stagnant_count = 0
            
            msg = f"[StopController Round {current_round}] n*推进: {self.best_n_star-gain} -> {n_star} (+{gain})"
            print(msg)
            
            # 检查是否达到上限
            if self.max_n and n_star >= self.max_n:
                reason = f"n*达到上限 ({n_star} >= {self.max_n})"
                print(f"[StopController] {reason}")
                return True, reason
            
            return False, ""
        
        elif n_star == self.best_n_star:
            # n*停滞
            self.stagnant_count += 1
            
            msg = f"[StopController Round {current_round}] n*停滞: {n_star} (连续{self.stagnant_count}轮)"
            print(msg)
            
            if self.stagnant_count >= self.patience:
                reason = f"n*停滞{self.stagnant_count}轮 (>= patience={self.patience})"
                print(f"[StopController] {reason}")
                return True, reason
            
            return False, ""
        
        else:
            # n*退步（罕见）
            self.stagnant_count += 1
            
            msg = f"[StopController Round {current_round}] 警告：n*退步: {self.best_n_star} -> {n_star}"
            print(msg)
            
            if self.stagnant_count >= self.patience:
                reason = f"n*退步且停滞{self.stagnant_count}轮"
                print(f"[StopController] {reason}")
                return True, reason
            
            return False, ""
    
    def reset(self):
        """重置控制器"""
        self.stagnant_count = 0
        self.best_n_star = 0
        self.n_star_history = []
        print("[StopController] 控制器已重置")
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'best_n_star': self.best_n_star,
            'stagnant_count': self.stagnant_count,
            'n_star_history': self.n_star_history,
            'patience_remaining': self.patience - self.stagnant_count
        }


class RoundLogger:
    """
    Round级别日志记录器
    
    功能：
    - 记录每个round的关键指标
    - 生成训练报告
    """
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: 日志文件路径
        """
        self.log_path = log_path
        self.rounds = []
    
    def log_round(
        self,
        round_id: int,
        metrics: Dict
    ):
        """
        记录一个round的信息
        
        Args:
            round_id: 轮次ID
            metrics: 指标字典
        """
        import numpy as np
        
        # 转换numpy类型为Python原生类型（避免后续JSON序列化错误）
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        metrics['round_id'] = round_id
        metrics_native = convert_to_native(metrics)
        self.rounds.append(metrics_native)
        
        # 追加写入日志文件
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Round {round_id}\n")
            f.write(f"{'='*80}\n")
            
            for key, value in sorted(metrics.items()):
                if key != 'round_id':
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    elif isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for k, v in value.items():
                            f.write(f"    {k}: {v}\n")
                    elif isinstance(value, list):
                        f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write(f"{'='*80}\n")
    
    def save_summary(self, summary_path: str):
        """
        保存训练摘要
        
        Args:
            summary_path: 摘要保存路径
        """
        import json
        import numpy as np
        
        def convert_to_json_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                # 通用numpy标量类型处理
                return obj.item()
            elif hasattr(obj, '__float__') and type(obj).__module__ == 'numpy':
                # 处理所有numpy浮点类型
                return float(obj)
            elif hasattr(obj, '__int__') and type(obj).__module__ == 'numpy':
                # 处理所有numpy整数类型
                return int(obj)
            else:
                return obj
        
        # 转换所有numpy类型
        rounds_serializable = convert_to_json_serializable(self.rounds)
        
        with open(summary_path, 'w') as f:
            json.dump(rounds_serializable, f, indent=2)
        
        print(f"[RoundLogger] 训练摘要已保存: {summary_path}")


if __name__ == "__main__":
    print("蒸馏训练模块测试")
    print("="*60)
    
    # 测试StopController
    controller = StopController(patience=3, max_n=10)
    
    # 模拟n*变化
    n_star_sequence = [3, 4, 5, 5, 5, 6, 6, 6, 6, 7]
    
    for round_id, n_star in enumerate(n_star_sequence):
        should_stop, reason = controller.should_stop(n_star, current_round=round_id)
        
        if should_stop:
            print(f"\nRound {round_id}: 触发停止条件 - {reason}")
            break
    
    # 打印状态
    status = controller.get_status()
    print(f"\n最终状态:")
    print(f"  best_n_star: {status['best_n_star']}")
    print(f"  stagnant_count: {status['stagnant_count']}")
    print(f"  n_star_history: {status['n_star_history']}")
    
    print("\n模块测试完成！")


