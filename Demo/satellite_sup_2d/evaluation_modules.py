"""
分层评估模块 - 按源项数量n评估teacher合格性

模块功能：
1. create_anchor_set: 为每个component_num创建anchor评估集
2. AnchorEvaluator: 按n分层评估RMSE (m_C和m_CS)
3. EligibilityGate: 判定teacher合格集合T和可靠前沿n*
4. IntervalScheduler: 动态调整蒸馏区间[A,B]

设计思路：
- m_C(n): 只用predictor的RMSE (c_only=True)
- m_CS(n): 用predictor+super的RMSE (c_only=False)
- 合格条件: m_CS(n) <= τ 且 (m_C - m_CS) >= Δ
- 区间策略: 以n*为中心，宽度动态调整
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from Demo.satellite_sup_2d.utilizes_satellite import load_h5_dataset
from Demo.satellite_sup_2d.ablation_satellite import transform_U_channels
from Demo.satellite_sup_2d.data_loader_satellite import resolve_satellite_h5_path


def create_anchor_set(
    component_nums: List[int],
    samples_per_n: int = 50,
    target_U_channels: int = 16,
    empty_channel_value: float = 1.0,
    base_path: str = "/data/wqn/datasets/dataset_20251218_mc",
    seed: int = 42
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    为每个component_num创建anchor评估集（带真值标签）
    
    Args:
        component_nums: 元件数量列表，如[1,2,3,4,5,6,7,8,9,10]
        samples_per_n: 每个n抽取的样本数
        target_U_channels: 目标U通道数
        empty_channel_value: 空通道填充值
        base_path: 数据集基础路径
        seed: 随机种子
    
    Returns:
        anchor_dict: {n: {'G': ..., 'U': ..., 'T': ...}}
    """
    np.random.seed(seed)
    anchor_dict = {}
    
    print("\n" + "=" * 60)
    print("创建Anchor评估集（按component_num分层）")
    print("=" * 60)
    
    for n in component_nums:
        try:
            h5_path = resolve_satellite_h5_path(base_path, n)
            dataset = load_h5_dataset(h5_path)
            G = dataset['G']
            U = dataset['U']
            T = dataset['T']
            
            total_samples = G.shape[0]
            actual_samples = min(samples_per_n, total_samples)
            
            # 随机抽取样本
            idx = np.random.choice(total_samples, actual_samples, replace=False)
            
            G_anchor = G[idx]
            U_anchor = U[idx]
            T_anchor = T[idx]
            
            # 转换U通道
            U_anchor = transform_U_channels(U_anchor, target_U_channels, empty_channel_value)
            
            anchor_dict[n] = {
                'G': G_anchor,
                'U': U_anchor,
                'T': T_anchor
            }
            
            print(f"[n={n:2d}] Anchor集: {actual_samples}个样本, "
                  f"U通道: {U.shape[-1]} -> {target_U_channels}")
        
        except Exception as e:
            print(f"[错误] 无法加载 n={n}: {e}")
            continue
    
    print(f"\nAnchor集创建完成，共{len(anchor_dict)}个n值")
    print("=" * 60 + "\n")
    
    return anchor_dict


class AnchorEvaluator:
    """
    Anchor集分层评估器
    
    功能：
    - 按component_num（n）分层评估模型RMSE
    - 支持C-only模式（m_C）和C+S模式（m_CS）
    """
    
    def __init__(
        self,
        model: nn.Module,
        anchor_dict: Dict[int, Dict[str, np.ndarray]],
        normalizers: Dict[str, any],
        device: torch.device
    ):
        """
        Args:
            model: supredictor_list_windows模型
            anchor_dict: create_anchor_set创建的anchor集
            normalizers: {'G': g_norm, 'U': u_norm, 'T': t_norm}
            device: 评估设备
        """
        self.model = model
        self.anchor_dict = anchor_dict
        self.normalizers = normalizers
        self.device = device
    
    def evaluate_by_n(self, c_only: bool = False) -> Dict[int, float]:
        """
        按n评估RMSE
        
        Args:
            c_only: True=评估m_C (只用predictor), False=评估m_CS (predictor+super)
        
        Returns:
            {n: rmse_value}
        """
        self.model.eval()
        results = {}
        
        mode_str = "C-only" if c_only else "C+S"
        print(f"\n评估模式: {mode_str}")
        print("-" * 40)
        
        with torch.no_grad():
            for n in sorted(self.anchor_dict.keys()):
                data = self.anchor_dict[n]
                
                # 归一化
                G_norm = self.normalizers['G'].norm(data['G'])
                U_norm = self.normalizers['U'].norm(data['U'])
                T_norm = self.normalizers['T'].norm(data['T'])
                
                # 转tensor
                G_tensor = torch.as_tensor(G_norm, dtype=torch.float).to(self.device)
                U_tensor = torch.as_tensor(U_norm, dtype=torch.float).to(self.device)
                
                # 预测
                pred_norm = self.model(G_tensor, U_tensor, c_only=c_only)
                
                # 反归一化计算RMSE
                pred = self.normalizers['T'].back(pred_norm.cpu().numpy())
                T_true = data['T']
                
                rmse = np.sqrt(np.mean((pred - T_true) ** 2))
                results[n] = rmse
                
                print(f"  n={n:2d}: RMSE={rmse:.6f}")
        
        print("-" * 40)
        return results


class EligibilityGate:
    """
    Teacher合格判定器
    
    功能：
    - 根据m_C和m_CS判定哪些n可以作为teacher（集合T）
    - 计算可靠前沿n*（最大可蒸馏n）
    
    合格条件：
    1. m_CS(n) <= τ_rmse (绝对精度达标)
    2. (m_C(n) - m_CS(n)) >= δ_rmse (C+S显著优于C)
    """
    
    def __init__(
        self,
        tau_rmse: float = 0.05,
        delta_rmse: float = 0.01,
        relative_improvement: float = 0.1
    ):
        """
        Args:
            tau_rmse: C+S绝对RMSE阈值
            delta_rmse: C+S相比C的最小改进量（绝对值）
            relative_improvement: 相对改进比例 (可选，备用判据)
        """
        self.tau_rmse = tau_rmse
        self.delta_rmse = delta_rmse
        self.relative_improvement = relative_improvement
    
    def compute_eligible_set(
        self,
        m_C_dict: Dict[int, float],
        m_CS_dict: Dict[int, float]
    ) -> Tuple[List[int], int, Dict]:
        """
        计算可蒸馏集合T和可靠前沿n*
        
        Args:
            m_C_dict: {n: rmse_C}
            m_CS_dict: {n: rmse_CS}
        
        Returns:
            T: 可蒸馏集合（通过条件的n列表）
            n_star: 可靠前沿（最大可蒸馏n）
            metrics: 诊断信息
        """
        T = []
        diagnostics = []
        
        print("\n" + "=" * 60)
        print("Teacher合格判定")
        print("=" * 60)
        print(f"判定条件:")
        print(f"  1. m_CS(n) <= {self.tau_rmse:.4f}")
        print(f"  2. m_C(n) - m_CS(n) >= {self.delta_rmse:.4f}")
        print("-" * 60)
        
        for n in sorted(m_CS_dict.keys()):
            if n not in m_C_dict:
                continue
            
            m_CS = m_CS_dict[n]
            m_C = m_C_dict[n]
            
            # 条件1：C+S绝对精度达标
            cond1 = m_CS <= self.tau_rmse
            
            # 条件2：C+S显著优于C
            improvement = m_C - m_CS
            cond2 = improvement >= self.delta_rmse
            
            # 相对改进（可选诊断）
            rel_improvement = improvement / (m_C + 1e-10)
            
            is_eligible = cond1 and cond2
            
            if is_eligible:
                T.append(n)
            
            status = "✓ 合格" if is_eligible else "✗ 不合格"
            print(f"  n={n:2d}: m_C={m_C:.6f}, m_CS={m_CS:.6f}, "
                  f"Δ={improvement:.6f} ({rel_improvement*100:.1f}%) - {status}")
            
            diagnostics.append({
                'n': n,
                'm_C': m_C,
                'm_CS': m_CS,
                'improvement': improvement,
                'rel_improvement': rel_improvement,
                'cond1': cond1,
                'cond2': cond2,
                'eligible': is_eligible
            })
        
        n_star = max(T) if T else 0
        
        print("-" * 60)
        print(f"可蒸馏集合 T: {T}")
        print(f"可靠前沿 n*: {n_star}")
        print("=" * 60 + "\n")
        
        metrics = {
            'm_C': m_C_dict,
            'm_CS': m_CS_dict,
            'diagnostics': diagnostics
        }
        
        return T, n_star, metrics


class IntervalScheduler:
    """
    动态蒸馏区间[A,B]调度器
    
    策略：
    - 以n*为中心，宽度动态调整
    - n*推进时扩大宽度（探索更高n）
    - n*停滞时保持或缩小宽度（稳固当前区间）
    """
    
    def __init__(
        self,
        initial_width: int = 3,
        max_width: int = 5,
        min_width: int = 1
    ):
        """
        Args:
            initial_width: 初始区间宽度
            max_width: 最大区间宽度
            min_width: 最小区间宽度
        """
        self.width = initial_width
        self.max_width = max_width
        self.min_width = min_width
        self.stagnant_count = 0
        self.prev_n_star = 0
    
    def update_interval(
        self,
        T: List[int],
        n_star: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        更新蒸馏区间[A,B]
        
        Args:
            T: 可蒸馏集合
            n_star: 可靠前沿
        
        Returns:
            (A, B): 蒸馏区间，如果T为空则返回(None, None)
        """
        if not T:
            print("[IntervalScheduler] 可蒸馏集合为空，无法设置区间")
            return None, None
        
        # 检测n*推进
        if n_star > self.prev_n_star:
            # n*推进，扩大宽度
            self.width = min(self.width + 1, self.max_width)
            self.stagnant_count = 0
            print(f"[IntervalScheduler] n*推进 ({self.prev_n_star} -> {n_star}), "
                  f"宽度扩大至 {self.width}")
        elif n_star == self.prev_n_star:
            # n*停滞
            self.stagnant_count += 1
            if self.stagnant_count >= 3:
                # 连续3轮停滞，缩小宽度
                self.width = max(self.width - 1, self.min_width)
                print(f"[IntervalScheduler] n*停滞{self.stagnant_count}轮, "
                      f"宽度缩小至 {self.width}")
        else:
            # n*退步（罕见）
            print(f"[IntervalScheduler] 警告：n*退步 ({self.prev_n_star} -> {n_star})")
        
        self.prev_n_star = n_star
        
        # 计算[A, B]
        A = max(min(T), n_star - self.width + 1)
        B = min(max(T), n_star + 1)
        
        # 确保A <= B且在T内
        A = max(A, min(T))
        B = min(B, max(T))
        
        if A > B:
            A = B
        
        print(f"[IntervalScheduler] 蒸馏区间: [{A}, {B}], 宽度={B-A+1}")
        
        return A, B
    
    def reset(self):
        """重置调度器"""
        self.stagnant_count = 0
        self.prev_n_star = 0
        print("[IntervalScheduler] 调度器已重置")


def test_anchor_evaluator():
    """测试anchor评估流程"""
    print("\n" + "="*60)
    print("测试Anchor评估模块")
    print("="*60)
    
    # 创建anchor集
    component_nums = [1, 2, 3, 4, 5]
    anchor_dict = create_anchor_set(
        component_nums,
        samples_per_n=10,
        target_U_channels=16
    )
    
    print(f"\nAnchor集创建完成：{len(anchor_dict)}个n值")
    for n, data in anchor_dict.items():
        print(f"  n={n}: {data['G'].shape[0]}个样本")
    
    # 测试合格判定
    gate = EligibilityGate(tau_rmse=0.05, delta_rmse=0.01)
    
    # 模拟评估结果
    m_C = {1: 0.08, 2: 0.06, 3: 0.04, 4: 0.03, 5: 0.025}
    m_CS = {1: 0.03, 2: 0.025, 3: 0.02, 4: 0.015, 5: 0.012}
    
    T, n_star, metrics = gate.compute_eligible_set(m_C, m_CS)
    
    # 测试区间调度
    scheduler = IntervalScheduler(initial_width=2, max_width=4)
    
    for round_id in range(5):
        print(f"\n--- Round {round_id} ---")
        A, B = scheduler.update_interval(T, n_star)
        print(f"区间: [{A}, {B}]")
        
        # 模拟n*推进
        if round_id == 2:
            n_star = min(n_star + 1, 5)
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_anchor_evaluator()
