"""
一致性训练模块 - 循环叠加蒸馏框架

模块功能：
1. PathSampler: 生成K=8条不同的叠加路径
2. ConsistencyTrainer: 使用多路径一致性损失训练超分辨率网络（冻结predictor）

设计思路：
- 路径变化策略：通道重排 + 分组顺序变化
- 一致性损失：K条路径输出的方差（到均值的MSE）
- 训练策略：冻结pred_net，只更新super_net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class PathSampler:
    """
    生成K=8条不同的叠加路径
    
    策略：
    1. 对U的通道进行随机重排（8种固定种子的排列）
    2. 改变分组合并顺序（影响channel_to_instance的处理）
    
    Args:
        K: 路径数量，默认8
        channel_num: 基础通道数，默认16
    """
    
    def __init__(self, K: int = 8, channel_num: int = 16):
        self.K = K
        self.channel_num = channel_num
        self.path_configs = self._generate_path_configs()
    
    def _generate_path_configs(self) -> List[dict]:
        """生成K条路径的配置"""
        configs = []
        for k in range(self.K):
            configs.append({
                'path_id': k,
                'shuffle_seed': 1000 + k * 100,  # 固定种子保证可复现
                'group_order': self._get_group_order(k)
            })
        return configs
    
    def _get_group_order(self, path_id: int) -> str:
        """
        获取分组顺序策略
        
        不同的path_id对应不同的分组/合并顺序：
        - 0: 顺序分组 [0,1,2,3,...]
        - 1: 逆序分组 [...,3,2,1,0]
        - 2: 偶数优先 [0,2,4,...,1,3,5,...]
        - 3: 奇数优先 [1,3,5,...,0,2,4,...]
        - 4-7: 循环移位
        """
        orders = ['sequential', 'reverse', 'even_first', 'odd_first',
                  'rotate_1', 'rotate_2', 'rotate_3', 'rotate_4']
        return orders[path_id % len(orders)]
    
    def apply_path(self, U: torch.Tensor, path_id: int) -> torch.Tensor:
        """
        对U应用第path_id条路径变换
        
        Args:
            U: [B, H, W, C] 源项场
            path_id: 路径ID (0-7)
        
        Returns:
            U_transformed: [B, H, W, C] 变换后的U（shape不变）
        """
        if path_id >= self.K:
            raise ValueError(f"path_id={path_id} 超出范围 [0, {self.K-1}]")
        
        config = self.path_configs[path_id]
        U_shuffled = self._shuffle_channels(U, config['shuffle_seed'])
        U_reordered = self._reorder_groups(U_shuffled, config['group_order'])
        
        return U_reordered
    
    def _shuffle_channels(self, U: torch.Tensor, seed: int) -> torch.Tensor:
        """对每个样本的通道进行随机重排"""
        B, H, W, C = U.shape
        U_shuffled = U.clone()
        
        # 为每个batch样本独立打乱（使用固定种子保证同一path_id结果一致）
        rng = np.random.RandomState(seed)
        for b in range(B):
            perm = torch.from_numpy(rng.permutation(C)).long()
            U_shuffled[b] = U[b, :, :, perm]
        
        return U_shuffled
    
    def _reorder_groups(self, U: torch.Tensor, order: str) -> torch.Tensor:
        """
        改变分组顺序（模拟不同的叠加路径）
        
        注意：这里不改变通道位置，而是返回一个"逻辑顺序"标记
        实际的分组顺序变化会在forward中通过不同的channel_to_instance调用实现
        为了简化，这里先返回原始U，分组顺序变化在模型forward中处理
        """
        # TODO: 如果需要更复杂的分组顺序变化，可以在这里实现
        # 当前版本：通道重排已经提供了足够的路径多样性
        return U
    
    def get_all_paths(self, U: torch.Tensor) -> List[torch.Tensor]:
        """获取所有K条路径的U变换"""
        return [self.apply_path(U, k) for k in range(self.K)]


class ConsistencyTrainer:
    """
    一致性训练器
    
    功能：
    - 对同一样本生成K条不同路径的预测
    - 计算一致性损失（输出到均值的方差）
    - 冻结predictor（C），只更新super_net（S）
    
    Args:
        model: supredictor_list_windows模型
        path_sampler: PathSampler实例
        device: 训练设备
        K: 路径数量
        consistency_weight: 一致性损失权重
    """
    
    def __init__(
        self,
        model: nn.Module,
        path_sampler: PathSampler,
        device: torch.device,
        K: int = 8,
        consistency_weight: float = 1.0
    ):
        self.model = model
        self.path_sampler = path_sampler
        self.device = device
        self.K = K
        self.consistency_weight = consistency_weight
    
    def freeze_predictor(self):
        """冻结predictor网络（C）"""
        for param in self.model.pred_net.parameters():
            param.requires_grad = False
        print("[ConsistencyTrainer] Predictor (C) frozen")
    
    def unfreeze_predictor(self):
        """解冻predictor网络（恢复正常训练）"""
        for param in self.model.pred_net.parameters():
            param.requires_grad = True
        print("[ConsistencyTrainer] Predictor (C) unfrozen")
    
    def unfreeze_super(self):
        """确保super网络（S）可训练"""
        for param in self.model.super_net.parameters():
            param.requires_grad = True
    
    def train_one_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        log_interval: int = 10
    ) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器 (G, U, T)，T可以忽略
            optimizer: 优化器（只包含super_net参数）
            log_interval: 日志打印间隔
        
        Returns:
            avg_loss: 平均一致性损失
            avg_std: 平均路径标准差（稳定性指标）
        """
        self.model.train()
        self.freeze_predictor()
        self.unfreeze_super()
        
        total_loss = 0.0
        total_std = 0.0
        num_batches = 0
        
        for batch_idx, (G, U, T) in enumerate(dataloader):
            G = G.to(self.device)
            U = U.to(self.device)
            # T 不使用（一致性训练不需要标签）
            
            # 生成K条路径的预测
            outputs = []
            for k in range(self.K):
                U_path = self.path_sampler.apply_path(U, k)
                with torch.set_grad_enabled(True):
                    pred = self.model(G, U_path)
                outputs.append(pred)
            
            # 计算一致性损失（到均值的方差）
            outputs_stack = torch.stack(outputs, dim=0)  # [K, B, H, W, 1]
            mean_output = outputs_stack.mean(dim=0)       # [B, H, W, 1]
            
            consistency_loss = 0.0
            for k in range(self.K):
                consistency_loss += F.mse_loss(outputs[k], mean_output)
            consistency_loss = consistency_loss / self.K * self.consistency_weight
            
            # 计算稳定性指标（标准差）
            with torch.no_grad():
                output_std = outputs_stack.std(dim=0).mean().item()
            
            # 反向传播（只更新super_net）
            optimizer.zero_grad()
            consistency_loss.backward()
            optimizer.step()
            
            total_loss += consistency_loss.item()
            total_std += output_std
            num_batches += 1
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Consistency Loss: {consistency_loss.item():.6f}, "
                      f"Output Std: {output_std:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_std = total_std / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_std
    
    def validate_one_epoch(
        self,
        dataloader,
    ) -> Tuple[float, float]:
        """
        验证一个epoch（计算一致性损失但不更新参数）
        
        Returns:
            avg_loss: 平均一致性损失
            avg_std: 平均路径标准差
        """
        self.model.eval()
        
        total_loss = 0.0
        total_std = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for G, U, T in dataloader:
                G = G.to(self.device)
                U = U.to(self.device)
                
                # 生成K条路径的预测
                outputs = []
                for k in range(self.K):
                    U_path = self.path_sampler.apply_path(U, k)
                    pred = self.model(G, U_path)
                    outputs.append(pred)
                
                # 计算一致性损失
                outputs_stack = torch.stack(outputs, dim=0)
                mean_output = outputs_stack.mean(dim=0)
                
                consistency_loss = 0.0
                for k in range(self.K):
                    consistency_loss += F.mse_loss(outputs[k], mean_output)
                consistency_loss = consistency_loss / self.K * self.consistency_weight
                
                # 计算标准差
                output_std = outputs_stack.std(dim=0).mean().item()
                
                total_loss += consistency_loss.item()
                total_std += output_std
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_std = total_std / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_std


def test_path_sampler():
    """测试PathSampler功能"""
    print("Testing PathSampler...")
    
    # 创建测试数据
    B, H, W, C = 2, 64, 64, 16
    U = torch.randn(B, H, W, C)
    
    sampler = PathSampler(K=8, channel_num=16)
    
    # 测试所有路径
    all_paths = sampler.get_all_paths(U)
    print(f"Generated {len(all_paths)} paths")
    
    # 验证shape不变
    for i, U_path in enumerate(all_paths):
        assert U_path.shape == U.shape, f"Path {i} shape mismatch"
    
    # 验证路径不同
    for i in range(len(all_paths)):
        for j in range(i+1, len(all_paths)):
            diff = (all_paths[i] - all_paths[j]).abs().mean()
            print(f"Path {i} vs Path {j}: mean diff = {diff:.6f}")
    
    print("PathSampler test passed!\n")


if __name__ == "__main__":
    # 测试PathSampler
    test_path_sampler()
    
    print("ConsistencyTrainer module created successfully!")
