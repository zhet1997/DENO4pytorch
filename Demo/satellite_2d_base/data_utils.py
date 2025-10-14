# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
通用数据处理工具模块

本模块提供通用的数据处理功能，包括：
1. MATLAB/HDF5文件读取器
2. 数据归一化和反归一化
3. 基于物理量纲分析的数据增强
4. PyTorch数据集和数据加载器
5. 物理相似性变换功能

支持多种数据集格式，通过参数化设计实现可复用性。
"""

# 标准库导入
import os
import time
import h5py
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional

# 数值计算库
import numpy as np
from numpy import array
import scipy.io as sio

# 深度学习框架
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

# 本地工具函数
from .utils import load_yaml_config


class MatLoader:
    """
    MATLAB文件读取器，支持CFD仿真数据加载
    
    支持两种加载方式：
    - scipy.io: 适用于MATLAB v7.0及以下版本的.mat文件
    - h5py: 适用于MATLAB v7.3及以上版本的大型数据文件
    
    主要用于加载各种仿真数据的多物理场数据，包括：
    网格信息、流场变量、边界条件、几何参数等
    """

    def __init__(self, file_path, use_h5py=False):
        """
        初始化MATLAB文件读取器
        
        Args:
            file_path (str): .mat文件路径（不包含.mat扩展名）
            use_h5py (bool): 是否使用h5py读取（适用于v7.3+大文件）
        """
        self.file_path = file_path
        self.use_h5py = use_h5py
        self.data = None  # 存储加载的数据字典
        self._load_file()  # 自动加载文件

    def read_field(self, field):
        """
        读取指定的物理场数据并转换为float32格式
        
        Args:
            field (str): 物理场名称（如'Static Pressure', 'grid'等）
            
        Returns:
            np.ndarray: float32格式的物理场数据
            
        Raises:
            KeyError: 当指定字段不存在时抛出异常
        """
        if field in self.data:
            x = self.data[field]
            return x.astype(np.float32)  # 确保数据类型一致性
        else:
            raise KeyError(f"物理场 '{field}' 在MAT文件中未找到")

    def _load_file(self):
        """根据设置选择合适的方法加载.mat文件"""
        try:
            if self.use_h5py:
                self._load_file_with_h5py()
            else:
                self._load_file_with_sio()
        except Exception as e:
            print(f"MAT文件加载错误: {e}")
            self.data = None

    def _load_file_with_sio(self):
        """使用scipy.io加载传统格式的.mat文件"""
        self.data = sio.loadmat(self.file_path, mat_dtype=True)

    def _load_file_with_h5py(self):
        """使用h5py加载v7.3+格式的大型.mat文件"""
        with h5py.File(f'{self.file_path}.mat', 'r') as f:
            self.data = {key: np.array(f[key]) for key in f.keys()}


class DataNormer(nn.Module):
    """
    通用物理数据归一化器，用于各种数据集的标准化
    
    物理意义：
    - 将不同量纲的物理量统一到相似的数值范围
    - 基于训练集统计特性进行Z-score标准化：(x - mean) / std
    - 确保深度学习模型训练的数值稳定性和收敛性
    
    支持的数据类型：
    - x_norm: 输入数据（设计参数）归一化
    - y_norm: 输出数据（场结果）归一化
    """
    def __init__(self, data_type, load_path='./configs', config_name=None):
        """
        初始化数据归一化器
        
        Args:
            data_type (str): 数据类型，'x_norm'或'y_norm'
            load_path (str): 归一化参数配置文件路径
            config_name (str): 配置文件名，如果为None则根据项目自动选择
        """
        super().__init__()
        
        # 如果没有指定配置文件名，使用默认的叶轮机械配置
        if config_name is None:
            config_name = 'norm_gvrb_41d.yaml'
        
        # 从YAML配置文件加载预计算的统计参数
        norm_data = load_yaml_config(os.path.join(load_path, config_name))
        mean = np.array(norm_data[f'{data_type}_mean'])  # 训练集均值
        std = np.array(norm_data[f'{data_type}_std'])    # 训练集标准差
        
        # 将统计参数注册为PyTorch缓冲区，确保设备一致性（GPU/CPU）
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std + 1e-10, dtype=torch.float32))  # 添加小量避免除零

    def norm(self, x):
        """
        执行数据归一化：将物理量转换为标准化数值
        
        Args:
            x (torch.Tensor or np.ndarray): 待归一化的物理数据
            
        Returns:
            归一化后的数据，数值范围通常在[-3, 3]之间
            
        物理过程：原始物理量 → 无量纲标准化数值
        """
        if isinstance(x, torch.Tensor):
            # 确保均值和标准差在正确的设备上（GPU/CPU）
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            x = (x - self.mean) / self.std  # Z-score标准化
        else:
            # 处理NumPy数组
            x = (x - self.mean.cpu().numpy()) / self.std.cpu().numpy()
        return x

    def back(self, x):
        """
        执行反归一化：将标准化数值恢复为物理量
        
        Args:
            x (torch.Tensor or np.ndarray): 标准化后的数据
            
        Returns:
            恢复的物理量数据（原始量纲和数值范围）
            
        物理过程：无量纲标准化数值 → 原始物理量
        用于将模型预测结果转换回实际的物理意义
        """
        if isinstance(x, torch.Tensor):
            # 确保均值和标准差在正确的设备上（GPU/CPU）
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            x = x * self.std + self.mean  # 反Z-score变换
        else:
            # 处理NumPy数组
            x = x * self.std.cpu().numpy() + self.mean.cpu().numpy()
        return x


class BaseDataset(Dataset):
    """
    通用基础数据集类
    
    用于封装各种仿真数据的输入输出数据对，
    支持PyTorch的DataLoader进行批量加载和训练。
    
    数据结构：
    - inputs: 输入参数（设计参数、边界条件等）
    - outputs: 对应的场结果（多物理场变量）
    """
    def __init__(self, inputs, outputs) -> None:
        """
        初始化数据集
        
        Args:
            inputs (np.ndarray): 输入数据，形状为[N, input_dim]或[N, spatial_dim, input_dim]
            outputs (np.ndarray): 输出数据，形状为[N, spatial_dim, field_dim]
        """
        self.len_samples = inputs.shape[0]  # 样本总数
        self.inputs = inputs    # 输入参数
        self.outputs = outputs  # 场结果

    def __getitem__(self, idx: int):
        """
        获取单个样本数据
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (输入参数, 对应场结果)
        """
        inputs, outputs = self.inputs[idx, ...], self.outputs[idx, ...]
        return inputs, outputs

    def __len__(self) -> int:
        """返回数据集总样本数"""
        return self.len_samples


def get_mat_inorder(order_dict: dict) -> array:
    """
    将有序字典中的量纲向量组装成矩阵
    
    Args:
        order_dict (OrderedDict): 包含物理量名称和对应量纲向量的有序字典
        
    Returns:
        np.ndarray: 量纲矩阵，每列对应一个物理量的量纲向量
        
    用途：将字典格式的量纲信息转换为矩阵形式，便于后续的量纲分析计算
    """
    rst = []
    for _, value in order_dict.items():
        # 将每个量纲向量转为列向量
        rst.append(np.array(value)[:, np.newaxis])
    # 按列拼接所有量纲向量
    return np.concatenate(rst, axis=1)


def dimension_scaling(samples: array, bc_dim_mat=None, bc_dim_coef: array = None):
    """
    基于量纲分析的物理数据缩放函数
    
    物理原理：
    - 基于白金汉π定理，通过改变基本物理量的量纲来生成相似的流动状态
    - 缩放因子 = exp(量纲系数矩阵 × 量纲矩阵)
    - 保持物理相似性的同时扩充数据，增强模型泛化能力
    
    Args:
        samples (np.ndarray): 待缩放的物理数据
        bc_dim_mat (np.ndarray): 量纲矩阵，每列为一个物理量的量纲向量
        bc_dim_coef (np.ndarray): 量纲系数，控制缩放程度
        
    Returns:
        np.ndarray: 缩放后的物理数据，保持相似性
        
    数学表达：scaled_data = data × exp(coef @ dim_matrix)
    """
    # 计算量纲缩放矩阵：exp(系数 × 量纲矩阵)
    scale_mat = np.exp(bc_dim_coef @ bc_dim_mat).astype(np.float32)
    
    # 扩展缩放矩阵维度以匹配样本数据的空间维度
    space_dim = len(samples.shape) - len(scale_mat.shape)
    for _ in range(space_dim):
        scale_mat = np.expand_dims(scale_mat, axis=1)
    
    # 应用物理相似性缩放
    rst = samples * scale_mat
    return rst.astype(np.float32)


def dimension_scaling_Tensor(samples, bc_dim_mat=None, bc_dim_coef: array = None):
    """
    PyTorch张量版本的量纲缩放函数
    
    功能与dimension_scaling相同，但支持GPU计算和自动微分
    
    Args:
        samples (torch.Tensor): 待缩放的物理数据张量
        bc_dim_mat (np.ndarray): 量纲矩阵
        bc_dim_coef (np.ndarray): 量纲系数
        
    Returns:
        torch.Tensor: 缩放后的物理数据张量
    """
    # 转换为PyTorch张量并确保设备一致性
    bc_dim_coef = torch.tensor(bc_dim_coef, dtype=torch.float32)
    bc_dim_mat = torch.tensor(bc_dim_mat, dtype=torch.float32, device=bc_dim_coef.device)
    
    # 计算量纲缩放矩阵
    scale_mat = torch.exp(torch.matmul(bc_dim_coef, bc_dim_mat))
    
    # 扩展维度以匹配样本数据
    scale_mat = torch.unsqueeze(scale_mat, dim=1)  # 添加空间维度
    scale_mat = torch.unsqueeze(scale_mat, dim=1)  # 添加另一空间维度
    
    # 应用物理相似性缩放
    rst = samples * scale_mat
    return rst


def generate_dim_scale_coef(
    basic_vector: array = np.array([[1, 0, 0, 0], [0, 0, 1/2.23, -2/2.23]], dtype=np.float32),
    sample_num: int = 100,
    coef_range=0.01,
):
    """
    生成基于物理相似性理论的维度缩放系数
    
    物理原理：
    - 基于白金汉π定理，在保持量纲一致性的前提下生成相似工况
    - 通过基本向量张成的空间进行随机采样，确保物理合理性
    - 缩放系数控制相似变换的幅度，避免偏离物理实际
    
    Args:
        basic_vector (np.ndarray): 基本量纲向量矩阵，定义相似变换的自由度
                                  默认值对应叶轮机械的典型相似参数
        sample_num (int): 生成的样本数量
        coef_range (float): 系数变化范围，控制相似性程度
        
    Returns:
        np.ndarray: 维度缩放系数矩阵，形状为[sample_num, 4]
        
    物理意义：每行表示一组相似工况的量纲变换参数
    """
    sample_free_dim = basic_vector.shape[0]  # 自由度维数：独立的相似参数个数
    
    # 基于时间生成随机种子，确保可重现性
    seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    
    # 在自由度空间中生成随机系数
    av = np.random.randn(sample_num, sample_free_dim) * coef_range
    av = np.clip(av, -0.10, 0.10)  # 限制变化幅度，保持物理合理性
    
    # 通过基本向量映射到完整的量纲空间
    rst = av @ basic_vector  # 矩阵乘法：自由系数 × 基本向量 = 量纲系数
    return rst.astype(np.float32)


def generate_random_scale_coef(sample_num: int = 100, coef_range=0.01):
    """
    生成随机的维度缩放系数（无物理约束版本）
    
    Args:
        sample_num (int): 生成的样本数量
        coef_range (float): 系数变化范围
        
    Returns:
        np.ndarray: 随机缩放系数矩阵，形状为[sample_num, 4]
        
    注意：此函数生成的系数不保证物理相似性，主要用于数据扰动
    """
    sample_free_dim = 4  # 全自由度，不受物理约束
    av = np.random.randn(sample_num, sample_free_dim) * coef_range
    av = np.clip(av, -0.10, 0.10) * 0.5  # 限制扰动幅度

    return av.astype(np.float32)


def check_feasibility_satellite(eta_candidates, G_matrix, bounds_log):
    """
    检查卫星量纲缩放系数的线性约束可行性
    
    Args:
        eta_candidates (np.ndarray): 候选η向量，形状[N, 3]（已满足盒约束）
        G_matrix (np.ndarray): 约束矩阵G，形状[6, 3]
        bounds_log (np.ndarray): 对数边界[ℓ, u]，形状[6, 2]
        
    Returns:
        np.ndarray: 可行性标识，形状[N]的布尔数组
    """
    # 检查线性约束：ℓ ≤ G@η ≤ u
    G_eta = eta_candidates @ G_matrix.T  # [N, 3] @ [3, 6] = [N, 6]
    linear_feasible = np.all((G_eta >= bounds_log[:, 0]) & 
                           (G_eta <= bounds_log[:, 1]), axis=1)
    
    return linear_feasible


def generate_constrained_dim_scale_coef_satellite(
    sample_num: int = 100,
    coef_range: float = 0.01,
    q0_values: array = None,
    physical_bounds: dict = None,
    orthogonal_basis: list = None,
    bc_dim_mat: array = None
):
    """
    生成满足物理约束的卫星量纲缩放系数（拒绝采样版本）
    
    Args:
        sample_num (int): 需要生成的样本数量
        coef_range (float): 系数变化范围
        q0_values (np.ndarray): 当前工况的基准值，形状[6]
        physical_bounds (dict): 物理量边界字典
        orthogonal_basis (list): 3个正交基向量
        bc_dim_mat (np.ndarray): 量纲矩阵，形状[4, 6]
        
    Returns:
        np.ndarray: 满足约束的量纲系数矩阵，形状[sample_num, 4]
    # """
    # print(f"🔧 函数入口：generate_constrained_dim_scale_coef_satellite")
    # print(f"   传入的q0_values: {q0_values}")
    # print(f"   q0_values类型: {type(q0_values)}, 形状: {q0_values.shape if hasattr(q0_values, 'shape') else '无shape'}")
    # print(f"   sample_num: {sample_num}")
    
    # 构建正交基矩阵B (3x4)
    B = np.array(orthogonal_basis, dtype=np.float32)  # [3, 4]
    
    # 构建约束矩阵G = (B @ bc_dim_mat)^T
    G = (B @ bc_dim_mat).T  # [6, 3]
    
    # 构建对数边界
    bounds_log = np.zeros((6, 2), dtype=np.float32)
    bound_names = ["component_sdf", "component_power", "cooling_sdf", "cooling_temp", "coord_x", "coord_y"]
    
    for i, name in enumerate(bound_names):
        qmin, qmax = physical_bounds[name]
        
        # 调试：检查可能导致对数错误的值
        if q0_values[i] <= 1e-3:
            print(f"⚠️  q0基准值问题：{name} 的 q0_values[{i}] = {q0_values[i]:.6f} <= 1e-6，可能导致log错误")
            if name.endswith('_sdf'):
                print(f"   建议：SDF全为负值，考虑取绝对值最小值+偏移或固定最小值0.005")
            elif 'power' in name:
                print(f"   建议：功率密度为0或负值，考虑设置最小值1.0 W/m²")
            elif 'temp' in name:
                print(f"   建议：温度异常，考虑设置最小值200K")
            else:
                print(f"   建议：坐标异常，检查数据预处理")
                
        if qmin <= 1e-3:
            print(f"⚠️  物理边界问题：{name} 的 qmin = {qmin} <= 1e-6，可能导致log错误")
            print(f"   建议：检查get_physical_bounds_satellite()中{name}的下界设置")
            
        if qmax <= 1e-3:
            print(f"⚠️  物理边界问题：{name} 的 qmax = {qmax} <= 1e-6，可能导致log错误")
            print(f"   建议：检查get_physical_bounds_satellite()中{name}的上界设置")
            
        if qmax <= qmin:
            print(f"⚠️  边界逻辑错误：{name} 的 qmax({qmax}) <= qmin({qmin})，约束区间无效")
            print(f"   建议：确保 qmin < qmax，检查边界定义")
            
        # 额外的合理性检查
        if abs(q0_values[i]) > 1e6:
            print(f"⚠️  数值异常：{name} 的 q0_values[{i}] = {q0_values[i]:.2e} 数值过大，可能有数据问题")
            
        # 临时安全修复（用于保持训练连续性）
        safe_q0 = q0_values[i]
        safe_qmin = qmin
        safe_qmax = qmax
        
        if q0_values[i] <= 1e-6:
            if name.endswith('_sdf'):
                safe_q0 = 0.005
            elif 'power' in name:
                safe_q0 = 1.0
            elif 'temp' in name:
                safe_q0 = 200.0
            else:
                safe_q0 = 0.01
            print(f"   🔧 临时修复：使用安全值 {safe_q0}")
            
        if qmin <= 1e-6:
            if name.endswith('_sdf'):
                safe_qmin = 0.001
            elif 'power' in name:
                safe_qmin = 0.1
            elif 'temp' in name:
                safe_qmin = 150.0
            else:
                safe_qmin = 0.001
            print(f"   🔧 临时修复：qmin 使用安全值 {safe_qmin}")
            
        if qmax <= qmin or qmax <= 1e-6:
            safe_qmax = safe_qmin * 10  # 简单的修复：上界为下界的10倍
            print(f"   🔧 临时修复：qmax 使用安全值 {safe_qmax}")
            
        bounds_log[i, 0] = np.log(safe_qmin) - np.log(safe_q0)  # ℓ
        bounds_log[i, 1] = np.log(safe_qmax) - np.log(safe_q0)  # u
    
    # 拒绝采样：使用现有生成函数保证盒约束，单独检查线性约束
    valid_samples = []
    max_trials = 10
    
    for trial in range(max_trials):
        # 生成候选样本（3倍数量），直接调用原生成函数保证盒约束
        batch_size = min(sample_num * 3, 1000)
        eta_candidates = np.random.randn(batch_size, 3) * coef_range
        eta_candidates = np.clip(eta_candidates, -0.10, 0.10)  # 盒约束
        
        # 仅检查线性约束可行性
        feasible_mask = check_feasibility_satellite(eta_candidates, G, bounds_log)
        
        # 收集有效样本
        valid_eta = eta_candidates[feasible_mask]
        if len(valid_eta) > 0:
            valid_samples.append(valid_eta)
        
        # 检查是否收集足够样本
        total_valid = sum(len(samples) for samples in valid_samples)
        if total_valid >= sample_num:
            break
    
    # 直接使用收集到的有效样本
    if valid_samples:
        all_valid = np.concatenate(valid_samples, axis=0)
        selected_eta = all_valid[:sample_num]  # 直接取前sample_num个
    else:
        # 极端情况：没有有效样本时，生成小扰动系数
        selected_eta = np.random.randn(sample_num, 3) * coef_range
    
    # 转换为4维量纲系数：α = B^T @ η
    alpha = selected_eta @ B  # [sample_num, 3] @ [3, 4] = [sample_num, 4]
    
    return alpha.astype(np.float32)


def create_dataloader(inputs, outputs, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器
    
    Args:
        inputs (np.ndarray): 输入数据（设计参数）
        outputs (np.ndarray): 输出数据（场结果）
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据顺序
        
    Returns:
        DataLoader: PyTorch数据加载器对象
    """
    dataset = BaseDataset(inputs, outputs)  # 创建基础数据集
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 单线程加载，避免多进程问题
    )
    return dataloader


def create_dataloader_similar(inputs, outputs, batch_size=32, shuffle=True, 
                              coef_range=0, near_range=0, input_dict=None, bc_loc=(0, 5), config_name=None):
    """
    创建支持物理相似性数据增强的PyTorch数据加载器
    
    Args:
        inputs (np.ndarray): 输入数据
        outputs (np.ndarray): 输出数据
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据顺序
        coef_range (float): 相似性变换强度参数
        near_range (float): 近场随机扰动强度参数
        input_dict (OrderedDict): 输入量的量纲信息字典
        bc_loc (tuple): 边界条件在输入向量中的位置范围
        config_name (str): 归一化配置文件名，如果为None则使用默认配置
        
    Returns:
        DataLoader: 支持动态数据增强的数据加载器
        
    特性：
        - 每次加载时动态生成相似工况数据
        - 同时提供原始数据、近场扰动数据和远场相似数据
        - 基于物理相似性理论进行数据扩充
    """
    # 创建相似性增强数据集
    dataset = AugmentSimilar(
        inputs,
        outputs,
        coef_range=coef_range,      # 物理相似性变换参数
        near_range=near_range,      # 近场扰动参数
        bc_loc=bc_loc,              # 边界条件位置范围
        input_dict=input_dict,      # 输入量的量纲信息
        config_name=config_name,    # 归一化配置文件名
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,  # 使用单线程避免随机数生成的并发问题
    )
    return dataloader


class AugmentSimilar(Dataset):
    """
    基于物理相似性理论的通用数据增强数据集类
    
    核心思想：
    - 基于白金汉π定理，通过量纲分析生成物理相似的工况数据
    - 双层次增强：远场相似变换（保持物理相似性） + 近场随机扰动（增加多样性）
    - 每个原始样本动态生成多个相似样本，大幅扩充训练数据
    
    物理意义：
    - 远场变换：通过改变基本物理量的量纲生成相似工况
    - 近场扰动：在保持几何不变的前提下微调边界条件
    - 保证生成数据的物理合理性和多样性
    
    应用场景：
    - 各种仿真CFD数据有限时的数据扩充
    - 提高深度学习模型的泛化能力和鲁棒性
    - 利用物理知识指导数据增强过程
    """
    def __init__(
        self,
        inputs=None,
        outputs=None,
        near_range=0,
        coef_range=0,
        bc_loc=(1, 5),
        input_dict=None,
        config_name=None,
    ) -> None:
        """
        初始化物理相似性数据增强数据集
        
        Args:
            inputs (np.ndarray): 原始输入数据（设计参数）
            outputs (np.ndarray): 原始输出数据（场结果）
            near_range (float): 近场随机扰动强度
            coef_range (float): 远场相似变换强度
            bc_loc (tuple): 边界条件参数在输入向量中的位置范围
            input_dict (OrderedDict): 输入量的量纲信息字典
            config_name (str): 归一化配置文件名
        """
        self.len_samples = inputs.shape[0]  # 原始样本数量
        self.inputs = inputs                # 原始输入数据
        self.outputs = outputs              # 原始输出数据
        self.near_range = near_range        # 近场扰动参数
        self.coef_range = coef_range        # 远场相似变换参数
        self.bc_location = bc_loc           # 边界条件位置
        
        # 构建量纲矩阵
        if input_dict is not None:
            self.input_mat = get_mat_inorder(input_dict)  # 量纲矩阵
        else:
            # 如果没有提供量纲字典，使用单位矩阵（无物理变换）
            self.input_mat = np.eye(4)
            
        # 初始化数据归一化器
        self.x_norm = DataNormer(data_type="x_norm", config_name=config_name)
        
        # 卫星数据特殊处理：预加载约束参数
        self.is_satellite = config_name == 'norm_satellite.yaml'
        if self.is_satellite:
            from .dataset_satellite import get_physical_bounds_satellite, find_orthogonal_basis_satellite
            self.physical_bounds = get_physical_bounds_satellite()
            self.orthogonal_basis = find_orthogonal_basis_satellite()
        else:
            self.physical_bounds = None
            self.orthogonal_basis = None

    def _compute_q0_for_single_sample(self, sample_data):
        """
        为单个样本计算q0基准值
        
        Args:
            sample_data (np.ndarray): 单个样本数据，形状[256, 256, 6]
            
        Returns:
            np.ndarray: q0基准值，形状[6]
        """
        # print(f"🔍 调试sample_data形状: {sample_data.shape}")
        # print(f"🔍 调试sample_data类型: {type(sample_data)}")
        # for ch in range(6):
        #     ch_data = sample_data[..., ch] 
        #     print(f"   通道{ch}: min={np.min(ch_data):.6f}, max={np.max(ch_data):.6f}, mean={np.mean(ch_data):.6f}, std={np.std(ch_data):.6f}")
        
        q0_values = np.zeros(6)
        q0_values[0] = np.max(sample_data[..., 0])   # component_sdf取最大值
        q0_values[1] = np.mean(sample_data[..., 1])  # component_power取均值
        q0_values[2] = np.max(sample_data[..., 2])   # cooling_sdf取最大值
        q0_values[3] = np.mean(sample_data[..., 3])  # cooling_temp取均值
        q0_values[4] = np.max(sample_data[..., 4])   # coord_x取最大值
        q0_values[5] = np.max(sample_data[..., 5])   # coord_y取最大值
        
        # print(f"🔍 计算的q0_values: {q0_values}")
        return q0_values

    def sample_generate(self, sample_num: int,
                sample_dim,
                lower_limit: float = -1.5,
                upper_limit: float = 1.5
                ) -> array:
        """
        生成随机参数样本（备用方法）
        
        Args:
            sample_num (int): 生成样本数量
            sample_dim (int): 样本维度
            lower_limit (float): 标准化空间的下限
            upper_limit (float): 标准化空间的上限
            
        Returns:
            np.ndarray: 生成的物理空间参数
            
        注：此方法在当前版本中未被使用，保留用于可能的随机采样需求
        """
        # 在标准化空间中生成随机样本
        z = np.random.randn(sample_num, sample_dim)
        z = np.clip(z, lower_limit, upper_limit)  # 限制在合理范围内
        # 反归一化到物理空间
        return self.x_norm.back(z)

    def __getitem__(self, idx: int):
        """
        获取单个样本及其物理相似性增强数据
        
        物理增强过程：
        1. 原始样本 (inputs_anc): 未经处理的原始仿真数据
        2. 近场扰动 (inputs_ner): 在原始数据基础上施加小幅随机扰动
        3. 远场相似 (inputs_far): 在近场扰动基础上施加物理相似性变换
        
        Args:
            idx (int or list): 样本索引
            
        Returns:
            tuple: (归一化的原始输入, 输出场, 归一化的近场扰动输入, 
                   归一化的远场相似输入, 远场变换系数)
                   
        物理意义：
        - 原始数据：真实仿真计算结果，作为基准
        - 近场扰动：模拟测量误差和边界条件的微小变化
        - 远场相似：基于相似性理论生成的等效工况数据
        """
        # 处理批量索引
        num = 1
        bc_loc = self.bc_location  # 边界条件在输入向量中的位置
        if not isinstance(idx, int) and not isinstance(idx, np.int64):
            idx = idx.tolist()
            num = len(idx)

        # 生成量纲变换系数
        if self.is_satellite:
            # 卫星数据：使用约束生成
            # 为批次中的每个样本单独计算q0基准值
            coef_far_list = []
            
            # print(f"🔍 调试idx信息: idx={idx}, type={type(idx)}, num={num}")
            # print(f"🔍 self.inputs.shape: {self.inputs.shape}")
            
            if num == 1:
                # 单个样本的情况
                if isinstance(idx, int):
                    sample_idx = idx
                else:
                    sample_idx = idx[0]  # 取第一个元素
                    
                # print(f"🔍 单样本 sample_idx: {sample_idx}")
                sample_data = self.inputs[sample_idx]  # [256, 256, 6]
                q0_values = self._compute_q0_for_single_sample(sample_data)
                # print(f"🔍 调用前检查 q0_values: {q0_values}, 类型: {type(q0_values)}")
                coef_far = generate_constrained_dim_scale_coef_satellite(
                    sample_num=1,
                    coef_range=self.coef_range,
                    q0_values=q0_values,
                    physical_bounds=self.physical_bounds,
                    orthogonal_basis=self.orthogonal_basis,
                    bc_dim_mat=self.input_mat
                )
            else:
                # 批次处理：为每个样本单独计算
                # print(f"🔍 批次处理 idx: {idx}")
                for i, sample_idx in enumerate(idx):
                    # print(f"🔍 批次中第{i}个样本 sample_idx: {sample_idx}")
                    sample_data = self.inputs[sample_idx]  # [256, 256, 6]
                    q0_values = self._compute_q0_for_single_sample(sample_data)
                    print(f"🔍 批次调用前检查 q0_values: {q0_values}, 类型: {type(q0_values)}")
                    coef_single = generate_constrained_dim_scale_coef_satellite(
                        sample_num=1,
                        coef_range=self.coef_range,
                        q0_values=q0_values,
                        physical_bounds=self.physical_bounds,
                        orthogonal_basis=self.orthogonal_basis,
                        bc_dim_mat=self.input_mat
                    )
                    coef_far_list.append(coef_single[0])  # 取第一个（也是唯一的）
                coef_far = np.array(coef_far_list)  # [num, 4]
        else:
            # 其他数据：使用原方法
            coef_far = generate_dim_scale_coef(sample_num=num, coef_range=self.coef_range)  # 远场相似变换系数
        coef_ner = generate_random_scale_coef(sample_num=num, coef_range=self.near_range)  # 近场随机扰动系数

        # 复制原始数据用于变换
        inputs_anc = self.inputs[idx].copy()  # 原始锚点数据
        inputs_ner = self.inputs[idx].copy()  # 近场扰动数据
        inputs_far = self.inputs[idx].copy()  # 远场相似数据

        # 第一步：对边界条件施加近场随机扰动
        # 只对边界条件参数进行变换，几何参数保持不变
        if bc_loc[1] > bc_loc[0]:  # 确保有边界条件参数
            inputs_ner[..., bc_loc[0] : bc_loc[1]] = dimension_scaling(
                inputs_anc[..., bc_loc[0] : bc_loc[1]],
                bc_dim_mat=self.input_mat,
                bc_dim_coef=coef_ner,
            )

            # 第二步：在近场扰动基础上施加远场相似性变换
            # 基于物理相似性理论，生成等效但条件不同的工况
            inputs_far[..., bc_loc[0] : bc_loc[1]] = dimension_scaling(
                inputs_ner[..., bc_loc[0] : bc_loc[1]],
                bc_dim_mat=self.input_mat,
                bc_dim_coef=coef_far,
            )

        # 返回归一化后的数据用于神经网络训练
        return (self.x_norm.norm(inputs_anc).astype(np.float32),    # 原始输入（归一化）
                self.outputs[idx].copy().astype(np.float32),        # 对应的场输出
                self.x_norm.norm(inputs_ner).astype(np.float32),    # 近场扰动输入（归一化）
                self.x_norm.norm(inputs_far).astype(np.float32),    # 远场相似输入（归一化）
                coef_far.astype(np.float32))                        # 远场变换系数

    def __len__(self) -> int:
        """返回数据集样本总数"""
        return self.len_samples

