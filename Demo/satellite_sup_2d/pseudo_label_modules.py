"""
伪标签生成模块 - 离线HDF5 shards管理

模块功能：
1. UnlabeledSampler: 抽取无标签(G,U,n)样本（当前从现有数据集复用）
2. TeacherManager: 管理teacher模型快照
3. PseudoLabelBuilder: 生成伪标签（K路径均值）并写入HDF5
4. ShardManager: 管理HDF5 shards的版本与清理

设计思路：
- 伪标签=K条路径输出的均值（简化版，不计算权重）
- HDF5压缩存储（gzip）
- 版本管理：保留最近N轮shards
- 灵活数据源：当前复用现有数据集，后续可切换到大规模无标签库
"""

import os
import glob
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from Demo.satellite_sup_2d.evaluation_modules import load_h5_dataset, transform_U_channels
from Demo.satellite_sup_2d.data_loader_satellite import resolve_satellite_h5_path


class UnlabeledSampler:
    """
    无标签数据采样器
    
    功能：
    - 从数据集中抽取(G,U,n)样本（不使用T）
    - 当前版本：复用现有标注数据集（忽略T）
    - 未来版本：可切换到大规模无标签库
    """
    
    def __init__(
        self,
        component_nums: List[int],
        base_path: str = "/data/wqn/datasets/dataset_20251218_mc",
        samples_per_n: int = 1000,
        use_existing_dataset: bool = True,
        seed: int = 9999
    ):
        """
        Args:
            component_nums: 元件数量列表
            base_path: 数据集基础路径
            samples_per_n: 每个n抽取的样本数
            use_existing_dataset: True=从现有数据集提取, False=从无标签库提取（未来）
            seed: 随机种子
        """
        self.component_nums = component_nums
        self.base_path = base_path
        self.samples_per_n = samples_per_n
        self.use_existing_dataset = use_existing_dataset
        self.seed = seed
    
    def sample_for_interval(
        self,
        A: int,
        B: int,
        target_U_channels: int = 16,
        empty_channel_value: float = 1.0,
        avoid_train_indices: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        为蒸馏区间[A,B]抽取无标签样本
        
        Args:
            A, B: 蒸馏区间（component_num范围）
            target_U_channels: 目标U通道数
            empty_channel_value: 空通道填充值
            avoid_train_indices: 避免与训练集重复的索引字典 {n: indices_array}
        
        Returns:
            data_dict: {n: {'G': ..., 'U': ..., 'n': ...}}
        """
        np.random.seed(self.seed)
        data_dict = {}
        
        print("\n" + "=" * 60)
        print(f"抽取无标签样本：区间[{A}, {B}]")
        print("=" * 60)
        
        for n in range(A, B + 1):
            if n not in self.component_nums:
                print(f"[跳过] n={n} 不在component_nums中")
                continue
            
            try:
                h5_path = resolve_satellite_h5_path(self.base_path, n)
                dataset = load_h5_dataset(h5_path)
                G = dataset['G']
                U = dataset['U']
                # T不使用（伪标签生成不需要真值）
                
                total_samples = G.shape[0]
                
                # 避免与训练集重复（如果提供了avoid_train_indices）
                if avoid_train_indices and n in avoid_train_indices:
                    avoid_idx = avoid_train_indices[n]
                    all_idx = np.arange(total_samples)
                    available_idx = np.setdiff1d(all_idx, avoid_idx)
                    print(f"  n={n}: 避免{len(avoid_idx)}个训练样本，剩余{len(available_idx)}个可用")
                else:
                    available_idx = np.arange(total_samples)
                
                # 随机抽取
                actual_samples = min(self.samples_per_n, len(available_idx))
                idx = np.random.choice(available_idx, actual_samples, replace=False)
                
                G_unlabeled = G[idx]
                U_unlabeled = U[idx]
                
                # 转换U通道
                U_unlabeled = transform_U_channels(
                    U_unlabeled, 
                    target_U_channels, 
                    empty_channel_value
                )
                
                data_dict[n] = {
                    'G': G_unlabeled,
                    'U': U_unlabeled,
                    'n': n
                }
                
                print(f"  n={n}: {actual_samples}个样本, U通道: {U.shape[-1]} -> {target_U_channels}")
            
            except Exception as e:
                print(f"[错误] 无法加载 n={n}: {e}")
                continue
        
        total_samples_count = sum(data['G'].shape[0] for data in data_dict.values())
        print(f"\n抽取完成：共{len(data_dict)}个n值，{total_samples_count}个样本")
        print("=" * 60 + "\n")
        
        return data_dict


class TeacherManager:
    """
    Teacher模型快照管理器
    
    功能：
    - 保存每个round的teacher快照
    - 加载指定round的teacher用于伪标签生成
    - 快照信息追踪
    """
    
    def __init__(self, work_path: str):
        """
        Args:
            work_path: 工作目录路径
        """
        self.work_path = work_path
        self.teacher_dir = os.path.join(work_path, 'teachers')
        os.makedirs(self.teacher_dir, exist_ok=True)
    
    def save_teacher(
        self,
        model: nn.Module,
        round_id: int,
        extra_info: Optional[Dict] = None
    ) -> str:
        """
        保存teacher快照
        
        Args:
            model: 要保存的模型
            round_id: 轮次ID
            extra_info: 额外信息（如评估指标）
        
        Returns:
            teacher_path: 保存路径
        """
        teacher_path = os.path.join(self.teacher_dir, f'teacher_round_{round_id}.pth')
        
        checkpoint = {
            'model_state': model.state_dict(),
            'round_id': round_id,
            'timestamp': time.time(),
            'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, teacher_path)
        
        print(f"[TeacherManager] Teacher快照已保存: round_{round_id}")
        return teacher_path
    
    def load_teacher(
        self,
        model_template: nn.Module,
        round_id: int,
        device: torch.device
    ) -> nn.Module:
        """
        加载teacher快照
        
        Args:
            model_template: 模型模板（用于加载权重）
            round_id: 轮次ID
            device: 加载设备
        
        Returns:
            teacher: 加载的teacher模型（eval模式）
        """
        teacher_path = os.path.join(self.teacher_dir, f'teacher_round_{round_id}.pth')
        
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher快照不存在: {teacher_path}")
        
        checkpoint = torch.load(teacher_path, map_location=device)
        model_template.load_state_dict(checkpoint['model_state'])
        model_template.eval()
        
        print(f"[TeacherManager] Teacher快照已加载: round_{round_id}")
        if 'save_time' in checkpoint:
            print(f"  保存时间: {checkpoint['save_time']}")
        
        return model_template
    
    def list_teachers(self) -> List[Tuple[int, str]]:
        """
        列出所有teacher快照
        
        Returns:
            [(round_id, teacher_path), ...]
        """
        teacher_files = sorted(glob.glob(os.path.join(self.teacher_dir, 'teacher_round_*.pth')))
        teachers = []
        
        for path in teacher_files:
            filename = os.path.basename(path)
            round_id = int(filename.replace('teacher_round_', '').replace('.pth', ''))
            teachers.append((round_id, path))
        
        return teachers


class PseudoLabelBuilder:
    """
    伪标签生成器
    
    功能：
    - 使用teacher对无标签样本生成伪标签
    - 伪标签=K条路径输出的均值（简化版，不使用权重）
    - 写入HDF5 shard（gzip压缩）
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        path_sampler,
        normalizers: Dict[str, any],
        device: torch.device,
        K: int = 8
    ):
        """
        Args:
            teacher: teacher模型（已加载权重）
            path_sampler: PathSampler实例
            normalizers: {'G': ..., 'U': ..., 'T': ...}
            device: 计算设备
            K: 路径数量
        """
        self.teacher = teacher
        self.path_sampler = path_sampler
        self.normalizers = normalizers
        self.device = device
        self.K = K
    
    def generate_shard(
        self,
        unlabeled_data: Dict[int, Dict[str, np.ndarray]],
        shard_path: str,
        round_id: int,
        batch_size: int = 16
    ) -> Dict[str, any]:
        """
        生成伪标签并写入HDF5 shard
        
        Args:
            unlabeled_data: {n: {'G': ..., 'U': ..., 'n': ...}}
            shard_path: HDF5保存路径
            round_id: 轮次ID
            batch_size: 推理batch size
        
        Returns:
            stats: 统计信息
        """
        self.teacher.eval()
        
        G_all, U_all, T_pseudo_all, n_all = [], [], [], []
        
        print("\n" + "=" * 60)
        print(f"生成伪标签 (Round {round_id})")
        print("=" * 60)
        
        total_samples = 0
        
        with torch.no_grad():
            for n in sorted(unlabeled_data.keys()):
                data = unlabeled_data[n]
                G = data['G']
                U = data['U']
                num_samples = G.shape[0]
                
                print(f"  n={n}: {num_samples}个样本...", end='', flush=True)
                
                # 归一化
                G_norm = self.normalizers['G'].norm(G)
                U_norm = self.normalizers['U'].norm(U)
                
                # 分batch生成伪标签
                T_pseudo_list = []
                
                for i in range(0, num_samples, batch_size):
                    batch_G = G_norm[i:i+batch_size]
                    batch_U = U_norm[i:i+batch_size]
                    
                    # 转tensor
                    batch_G_tensor = torch.as_tensor(batch_G, dtype=torch.float).to(self.device)
                    batch_U_tensor = torch.as_tensor(batch_U, dtype=torch.float).to(self.device)
                    
                    # K条路径预测
                    outputs = []
                    for k in range(self.K):
                        U_path = self.path_sampler.apply_path(batch_U_tensor, k)
                        pred = self.teacher(batch_G_tensor, U_path)
                        outputs.append(pred.cpu().numpy())
                    
                    # 均值作为伪标签
                    batch_T_pseudo = np.mean(outputs, axis=0)
                    T_pseudo_list.append(batch_T_pseudo)
                
                T_pseudo = np.concatenate(T_pseudo_list, axis=0)
                
                G_all.append(G)
                U_all.append(U)
                T_pseudo_all.append(T_pseudo)
                n_all.extend([n] * num_samples)
                
                total_samples += num_samples
                print(f" 完成")
        
        # 合并所有数据
        G_combined = np.concatenate(G_all, axis=0)
        U_combined = np.concatenate(U_all, axis=0)
        T_pseudo_combined = np.concatenate(T_pseudo_all, axis=0)
        n_combined = np.array(n_all, dtype=np.int32)
        
        print(f"\n生成完成：共{total_samples}个伪标签样本")
        print(f"  G shape: {G_combined.shape}")
        print(f"  U shape: {U_combined.shape}")
        print(f"  T_pseudo shape: {T_pseudo_combined.shape}")
        print(f"  n shape: {n_combined.shape}")
        
        # 写入HDF5（gzip压缩）
        print(f"\n写入HDF5: {shard_path}")
        
        with h5py.File(shard_path, 'w') as f:
            f.create_dataset('G', data=G_combined, compression='gzip', compression_opts=4)
            f.create_dataset('U', data=U_combined, compression='gzip', compression_opts=4)
            f.create_dataset('T_pseudo', data=T_pseudo_combined, compression='gzip', compression_opts=4)
            f.create_dataset('n', data=n_combined, compression='gzip', compression_opts=4)
            
            # 元信息
            f.attrs['round_id'] = round_id
            f.attrs['timestamp'] = time.time()
            f.attrs['save_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.attrs['K_paths'] = self.K
            f.attrs['total_samples'] = total_samples
            f.attrs['interval_A'] = min(unlabeled_data.keys())
            f.attrs['interval_B'] = max(unlabeled_data.keys())
            f.attrs['n_values'] = str(sorted(unlabeled_data.keys()))
        
        # 文件大小
        file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")
        print("=" * 60 + "\n")
        
        stats = {
            'total_samples': total_samples,
            'file_size_mb': file_size_mb,
            'n_values': sorted(unlabeled_data.keys()),
            'shard_path': shard_path
        }
        
        return stats


class ShardManager:
    """
    HDF5 Shard管理器
    
    功能：
    - 管理shard路径
    - 版本清理（保留最近N轮）
    - 空间管理
    """
    
    def __init__(
        self,
        work_path: str,
        keep_last_n: int = 2
    ):
        """
        Args:
            work_path: 工作目录路径
            keep_last_n: 保留最近N轮shards
        """
        self.work_path = work_path
        self.shard_dir = os.path.join(work_path, 'pseudo_shards')
        os.makedirs(self.shard_dir, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def get_shard_path(self, round_id: int) -> str:
        """获取shard路径"""
        return os.path.join(self.shard_dir, f'pseudo_round_{round_id}.h5')
    
    def cleanup_old_shards(self, current_round: int):
        """
        清理旧版本shards
        
        Args:
            current_round: 当前轮次ID
        """
        all_shards = sorted(glob.glob(os.path.join(self.shard_dir, 'pseudo_round_*.h5')))
        
        if len(all_shards) <= self.keep_last_n:
            print(f"[ShardManager] Shard数量({len(all_shards)}) <= keep_last_n({self.keep_last_n})，无需清理")
            return
        
        print(f"\n[ShardManager] 清理旧shards (保留最近{self.keep_last_n}轮)...")
        
        deleted_count = 0
        deleted_size_mb = 0.0
        
        for shard_path in all_shards:
            filename = os.path.basename(shard_path)
            round_id = int(filename.replace('pseudo_round_', '').replace('.h5', ''))
            
            # 删除旧于keep_last_n轮的shards
            if round_id < current_round - self.keep_last_n + 1:
                file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
                os.remove(shard_path)
                deleted_count += 1
                deleted_size_mb += file_size_mb
                print(f"  删除: round_{round_id} ({file_size_mb:.2f} MB)")
        
        if deleted_count > 0:
            print(f"清理完成：删除{deleted_count}个shards，释放{deleted_size_mb:.2f} MB空间\n")
        else:
            print("无需清理\n")
    
    def list_shards(self) -> List[Tuple[int, str, float]]:
        """
        列出所有shards
        
        Returns:
            [(round_id, shard_path, size_mb), ...]
        """
        shard_files = sorted(glob.glob(os.path.join(self.shard_dir, 'pseudo_round_*.h5')))
        shards = []
        
        for path in shard_files:
            filename = os.path.basename(path)
            round_id = int(filename.replace('pseudo_round_', '').replace('.h5', ''))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            shards.append((round_id, path, size_mb))
        
        return shards
    
    def get_total_size_mb(self) -> float:
        """获取所有shards的总大小(MB)"""
        shards = self.list_shards()
        return sum(size for _, _, size in shards)


if __name__ == "__main__":
    print("伪标签生成模块测试")
    print("="*60)
    
    # 测试UnlabeledSampler
    sampler = UnlabeledSampler(
        component_nums=[1, 2, 3, 4, 5],
        samples_per_n=20
    )
    
    data_dict = sampler.sample_for_interval(A=2, B=4, target_U_channels=16)
    
    print(f"\n抽取样本：{len(data_dict)}个n值")
    for n, data in data_dict.items():
        print(f"  n={n}: G={data['G'].shape}, U={data['U'].shape}")
    
    print("\n模块测试完成！")
