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
卫星热仿真数据处理模块

本模块用于处理卫星热仿真数据，主要功能包括：
1. 从h5py文件加载卫星热传导数据
2. 统一数据归一化和反归一化处理
3. 基于物理量纲分析的数据增强技术
4. 为深度学习模型准备训练和测试数据集
5. 支持基于热传导相似性理论的数据扩充

物理背景：
- 处理卫星内部的复杂三维热传导场
- 涉及多个物理量：元件SDF、功率密度、散热窗SDF、温度、坐标
- 基于白金汉π定理进行量纲分析和相似性变换
"""

# 标准库导入
import os
import time
import h5py
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional
import warnings

# 数值计算库
import numpy as np
from numpy import array
import sklearn.model_selection

# 深度学习框架
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

# 本地工具函数
from .utils import load_yaml_config
from .data_utils import (
    DataNormer, BaseDataset, AugmentSimilar,
    create_dataloader, create_dataloader_similar,
    dimension_scaling, get_mat_inorder,
    generate_dim_scale_coef
)


def load_satellite_data(data_path: str = "./data_post/heat_dataset_new.h5", sample_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从h5py格式文件加载卫星热仿真数据
    
    Args:
        data_path (str): h5文件路径
        sample_limit (Optional[int]): 限制加载的样本数量，None表示加载全部数据
        
    Returns:
        tuple: (输入数据, 输出数据)
               输入：形状为[N, 256, 256, 6] (component_sdf, component_power, cooling_sdf, 
                                         cooling_temp, coord_x, coord_y)
               输出：形状为[N, 256, 256, 1] (temperature)
               
    数据结构：
        - inputs通道0: 元件SDF (signed distance field)
        - inputs通道1: 元件功率密度 (W/m²)
        - inputs通道2: 散热窗SDF 
        - inputs通道3: 散热窗温度 (K)
        - inputs通道4: 坐标X (m) - 以中心为原点
        - inputs通道5: 坐标Y (m) - 以中心为原点
        - outputs通道0: 温度场 (K)
    """
    try:
        with h5py.File(data_path, 'r') as f:
            # 读取数据
            inputs = np.array(f['inputs'], dtype=np.float32)
            outputs = np.array(f['outputs'], dtype=np.float32)
            
            # 检查数据维度
            expected_input_shape = (None, 256, 256, 6)
            expected_output_shape = (None, 256, 256, 1)
            
            if inputs.ndim != 4 or inputs.shape[1:] != (256, 256, 6):
                raise ValueError(f"输入数据形状错误: {inputs.shape}, 期望: {expected_input_shape}")
            if outputs.ndim != 4 or outputs.shape[1:] != (256, 256, 1):
                raise ValueError(f"输出数据形状错误: {outputs.shape}, 期望: {expected_output_shape}")
            
            # 限制数据量（用于调试）
            total_samples = inputs.shape[0]
            if sample_limit is not None and sample_limit < total_samples:
                inputs = inputs[:sample_limit]
                outputs = outputs[:sample_limit]
                print(f"✅ 成功加载卫星数据（调试模式）:")
                print(f"   - 总样本数量: {total_samples}")
                print(f"   - 实际加载: {sample_limit} 样本")
            else:
                print(f"✅ 成功加载卫星数据:")
                print(f"   - 样本数量: {inputs.shape[0]}")
                
            print(f"   - 输入形状: {inputs.shape}")
            print(f"   - 输出形状: {outputs.shape}")
            print(f"   - 输入通道: component_sdf, component_power, cooling_sdf, cooling_temp, coord_x, coord_y")
            print(f"   - 输出通道: temperature")
            
            return inputs, outputs
            
    except Exception as e:
        raise RuntimeError(f"加载卫星数据失败: {e}")


def get_bc_dict_satellite() -> Tuple[OrderedDict, OrderedDict]:
    """
    获取卫星数据集的量纲分析字典
    
    基于白金汉π定理进行量纲分析，用于物理相似性变换。
    每个物理量用4个基本量纲的指数表示：[M, L, T, Θ]
    - M: 质量量纲 (Mass)
    - L: 长度量纲 (Length) 
    - T: 时间量纲 (Time)
    - Θ: 温度量纲 (Temperature)
    
    Returns:
        tuple: (输入量纲字典, 输出量纲字典)
    """
    # 输入参数的量纲矩阵（所有6个通道都参与变换，保持物理一致性）
    input_dict = OrderedDict([
        ("component_sdf", [0, 1, 0, 0]),      # 长度 [L] - SDF以米为单位
        ("component_power", [1, 0, -3, 0]),   # 功率密度 [M T⁻³] - W/m² = kg/s³
        ("cooling_sdf", [0, 1, 0, 0]),        # 长度 [L] - SDF以米为单位
        ("cooling_temp", [0, 0, 0, 1]),       # 温度 [Θ] - K
        ("coord_x", [0, 1, 0, 0]),            # 长度 [L] - X坐标以米为单位
        ("coord_y", [0, 1, 0, 0]),            # 长度 [L] - Y坐标以米为单位
    ])
    
    # 输出变量的量纲矩阵
    output_dict = OrderedDict([
        ("temperature", [0, 0, 0, 1]),         # 温度 [Θ] - K
    ])
    
    return input_dict, output_dict


def get_physical_bounds_satellite() -> Dict[str, Tuple[float, float]]:
    """
    获取卫星数据集各物理量q0基准值的合理取值范围
    
    注意：这里的边界是针对q0基准值的约束，不是原始数据分布
    - SDF类型：使用最大值作为q0，边界反映"最大值"的可能范围  
    - 其他类型：使用均值作为q0，边界反映"均值"的可能范围
    - 所有边界必须为正数（用于对数约束）
    
    Returns:
        Dict: 物理量名称 -> (下界, 上界) 的映射
    """
    return {
        # component_sdf最大值：SDF最大值通常在0.01~0.20范围（距离物体最远点）
        "component_sdf": (0.01, 0.20),        
        
        # component_power均值：基于统计mean=279, std=582，均值通常在50~1500范围
        "component_power": (100.0, 1500.0),     
        
        # cooling_sdf最大值：散热SDF最大值通常在0.01~1.5范围  
        "cooling_sdf": (0.01, 1.00),          
        
        # cooling_temp均值：基于统计mean=274, std=31，均值通常在200~350K范围
        "cooling_temp": (250.0, 300.0),
        
        # coord_x均值：坐标范围通常在-0.5~0.5m（以中心为原点），均值接近0但要为正
        "coord_x": (0.25, 0.6),
        
        # coord_y均值：坐标范围通常在-0.5~0.5m（以中心为原点），均值接近0但要为正  
        "coord_y": (0.25, 0.6),
    }


def find_orthogonal_basis_satellite(cache_file="./satellite_orthogonal_basis.npy") -> List[np.ndarray]:
    """
    计算卫星数据集的量纲零空间正交基（带文件缓存功能）
    
    基于导热率量纲约束：κ [M L² T⁻³ Θ⁻¹] = constant (二维导热问题)
    使用SVD分解寻找满足约束的3个自由度基向量
    
    Args:
        cache_file (str): 缓存文件路径，存储计算好的正交基以供复用
    
    Returns:
        List[np.ndarray]: 3个正交基向量，每个为[α_M, α_L, α_t, α_T]
        
    物理意义：
        - 基向量定义了在保持导热率不变前提下的相似变换自由度
        - 3个自由度对应3种独立的物理相似变换模式
    """
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        try:
            cached_basis = np.load(cache_file, allow_pickle=True)
            if len(cached_basis) == 3 and all(vec.shape == (4,) for vec in cached_basis):
                print(f"✅ 从缓存文件加载正交基: {cache_file}")
                # 仍然打印基向量信息
                dim_labels = ["α_M（质量）", "α_L（长度）", "α_t（时间）", "α_T（温度）"]
                for i, vec in enumerate(cached_basis):
                    rounded = [float(x) for x in vec.round(4)]
                    print(f"基{i+1}: {rounded} → {dict(zip(dim_labels, rounded))}")
                return cached_basis.tolist()
        except Exception as e:
            print(f"⚠️  缓存文件读取失败: {e}，重新计算正交基")
    
    # 重新计算正交基
    # 二维导热率的量纲向量：κ = [M¹ L² T⁻³ Θ⁻¹]
    kappa_dim = np.array([1, 2, -3, -1], dtype=np.float32)
    
    A = kappa_dim.reshape(1, -1)
    _, _, Vh = np.linalg.svd(A)
    null_space = Vh[1:].T

    alpha1, alpha2, alpha3 = null_space[:, 0], null_space[:, 1], null_space[:, 2]

    # 验证正交性
    assert np.isclose(np.dot(alpha1, kappa_dim), 0, atol=1e-5), f"基1不满足κ约束：点积={np.dot(alpha1, kappa_dim):.6f}"
    assert np.isclose(np.dot(alpha2, kappa_dim), 0, atol=1e-5), f"基2不满足κ约束：点积={np.dot(alpha2, kappa_dim):.6f}"
    assert np.isclose(np.dot(alpha3, kappa_dim), 0, atol=1e-5), f"基3不满足κ约束：点积={np.dot(alpha3, kappa_dim):.6f}"

    # 正交化+归一化
    alpha2 = alpha2 - (np.dot(alpha2, alpha1) / np.dot(alpha1, alpha1)) * alpha1
    alpha3 = alpha3 - (np.dot(alpha3, alpha1) / np.dot(alpha1, alpha1)) * alpha1 - \
             (np.dot(alpha3, alpha2) / np.dot(alpha2, alpha2)) * alpha2
    alpha1, alpha2, alpha3 = [vec / np.linalg.norm(vec) for vec in [alpha1, alpha2, alpha3]]

    # 保存到缓存文件
    try:
        basis_array = np.array([alpha1, alpha2, alpha3])
        np.save(cache_file, basis_array)
        print(f"✅ 正交基已保存到缓存文件: {cache_file}")
    except Exception as e:
        print(f"⚠️  缓存文件保存失败: {e}")

    # 打印基向量（控制台）
    print("✅ 正交基生成完成（二维导热问题，确保κ不变）：")
    dim_labels = ["α_M（质量）", "α_L（长度）", "α_t（时间）", "α_T（温度）"]
    for vec, name in zip([alpha1, alpha2, alpha3], ["基1", "基2", "基3"]):
        rounded = [float(x) for x in vec.round(4)]
        print(f"{name}: {rounded} → {dict(zip(dim_labels, rounded))}")
    return [alpha1, alpha2, alpha3]




def load_satellite_dataset(config: DictConfig, shuffle=True, sample_limit: Optional[int] = None) -> Tuple[DataLoader, List[DataLoader]]:
    """
    根据配置文件加载卫星标准监督数据集
    
    Args:
        config (DictConfig): 包含数据路径、样本数量等配置的字典
        shuffle (bool): 是否打乱训练数据顺序
        sample_limit (Optional[int]): 限制加载的样本数量，None表示加载全部数据
        
    Returns:
        tuple: (训练数据加载器, 测试数据加载器列表)
        
    数据处理流程：
        1. 从h5文件加载原始数据
        2. 按8:2随机分割训练/测试集
        3. 应用数据归一化（坐标可选）
        4. 将测试集分成4个子集以便批量验证
    """
    # 加载原始数据
    data_path = getattr(config.data, 'path', './data_post/heat_dataset_new.h5')
    inputs, outputs = load_satellite_data(data_path, sample_limit=sample_limit)
    
    # 获取数据集配置
    train_split = getattr(config.data, 'train_split', 0.8)
    
    # 随机分割数据集（8:2）
    total_samples = inputs.shape[0]
    train_size = int(total_samples * train_split)
    
    # 使用sklearn进行随机分割，确保可重现性
    train_indices, test_indices = sklearn.model_selection.train_test_split(
        np.arange(total_samples), 
        train_size=train_size, 
        random_state=42,
        shuffle=True
    )
    
    # 分割数据
    input_tensor_train = inputs[train_indices].astype(np.float32)
    output_tensor_train = outputs[train_indices].astype(np.float32)
    input_tensor_test = inputs[test_indices].astype(np.float32)
    output_tensor_test = outputs[test_indices].astype(np.float32)
    
    print(f"✅ 数据集分割完成:")
    print(f"   - 训练集: {input_tensor_train.shape[0]} 样本")
    print(f"   - 测试集: {input_tensor_test.shape[0]} 样本")
    print(f"   - 归一化模式: 统一归一化（所有通道）")
    
    # 应用归一化
    if hasattr(config.data, 'apply_normalization') and config.data.apply_normalization:
        x_norm = DataNormer(data_type='x_norm', config_name='norm_satellite.yaml')
        y_norm = DataNormer(data_type='y_norm', config_name='norm_satellite.yaml')
        
        input_tensor_train = x_norm.norm(input_tensor_train)
        output_tensor_train = y_norm.norm(output_tensor_train)
        input_tensor_test = x_norm.norm(input_tensor_test)
        output_tensor_test = y_norm.norm(output_tensor_test)
        print("✅ 应用统一数据归一化（所有通道）")
    
    # 创建训练数据加载器
    data_loader_train = create_dataloader(
        input_tensor_train,
        output_tensor_train,
        batch_size=getattr(config.train, 'batch_size', 32),
        shuffle=shuffle,
    )
    
    # 将测试数据分成4个子集，便于批量验证和内存管理
    input_test_lst = np.array_split(input_tensor_test, 4)
    output_test_lst = np.array_split(output_tensor_test, 4)
    data_loader_test = [create_dataloader(
        input_test,
        output_test,
        batch_size=getattr(config.train, 'batch_size', 32),
        shuffle=False,  # 测试集不打乱
    ) for input_test, output_test in zip(input_test_lst, output_test_lst)]
    
    return data_loader_train, data_loader_test


def load_satellite_dataset_similar(config: DictConfig, shuffle=True, sample_limit: Optional[int] = None) -> Tuple[DataLoader, List[DataLoader]]:
    """
    加载支持物理相似性数据增强的卫星训练数据集
    
    相比load_satellite_dataset函数，此函数额外支持：
    - 基于热传导相似性理论的动态数据增强
    - 三自由度相似变换（基于正交基）
    - 双层次扰动：远场相似变换 + 近场随机扰动
    - 在线数据生成，提供更丰富的训练样本
    
    Args:
        config (DictConfig): 配置参数，包含相似性和扰动范围设置
        shuffle (bool): 是否打乱数据顺序
        sample_limit (Optional[int]): 限制加载的样本数量，None表示加载全部数据
        
    Returns:
        tuple: (相似性增强的训练数据加载器, 标准测试数据加载器列表)
    """
    # 加载原始数据
    data_path = getattr(config.data, 'path', './data_post/heat_dataset_new.h5')
    inputs, outputs = load_satellite_data(data_path, sample_limit=sample_limit)
    
    # 获取数据集配置
    train_split = getattr(config.data, 'train_split', 0.8)
    coef_range = getattr(config.data, 'similarity', 0.01)    # 相似性变换强度
    near_range = getattr(config.data, 'near', 0.01)         # 近场扰动强度
    
    # 随机分割数据集
    total_samples = inputs.shape[0]
    train_size = int(total_samples * train_split)
    
    train_indices, test_indices = sklearn.model_selection.train_test_split(
        np.arange(total_samples), 
        train_size=train_size, 
        random_state=42,
        shuffle=True
    )
    
    # 分割数据
    input_tensor_train = inputs[train_indices].astype(np.float32)
    output_tensor_train = outputs[train_indices].astype(np.float32)
    input_tensor_test = inputs[test_indices].astype(np.float32)
    output_tensor_test = outputs[test_indices].astype(np.float32)
    
    print(f"✅ 相似性增强数据集准备:")
    print(f"   - 训练集: {input_tensor_train.shape[0]} 样本")
    print(f"   - 测试集: {input_tensor_test.shape[0]} 样本")
    print(f"   - 相似性变换强度: {coef_range}")
    print(f"   - 近场扰动强度: {near_range}")
    print(f"   - 归一化模式: 统一归一化（所有通道）")
    
    # 预处理：只对输出数据归一化，输入数据在AugmentSimilar中处理
    y_norm = DataNormer(data_type='y_norm', config_name='norm_satellite.yaml')
    if hasattr(config.data, 'apply_normalization') and config.data.apply_normalization:
        output_tensor_train = y_norm.norm(output_tensor_train)
        output_tensor_test = y_norm.norm(output_tensor_test)
    
    # 获取卫星数据的量纲信息
    input_dict, _ = get_bc_dict_satellite()
    
    # 创建支持相似性增强的训练数据加载器
    data_loader_train = create_dataloader_similar(
        input_tensor_train,
        output_tensor_train,
        batch_size=getattr(config.train, 'batch_size', 32),
        shuffle=shuffle,
        coef_range=coef_range,      # 物理相似性变换参数
        near_range=near_range,      # 近场扰动参数
        input_dict=input_dict,      # 卫星数据量纲字典
        bc_loc=(0, 6),              # 所有6个通道都参与物理变换，保持空间尺度一致性
        config_name='norm_satellite.yaml'  # 关键：指定卫星归一化配置
    )
    
    # 创建标准的测试数据加载器（分4个子集）
    x_norm = DataNormer(data_type='x_norm', config_name='norm_satellite.yaml')
    if hasattr(config.data, 'apply_normalization') and config.data.apply_normalization:
        input_tensor_test = x_norm.norm(input_tensor_test)
        
    input_test_lst = np.array_split(input_tensor_test, 4)
    output_test_lst = np.array_split(output_tensor_test, 4)
    data_loader_test = [create_dataloader(
        input_test,
        output_test,
        batch_size=getattr(config.train, 'batch_size', 32),
        shuffle=False,
    ) for input_test, output_test in zip(input_test_lst, output_test_lst)]
    
    return data_loader_train, data_loader_test

class SatelliteSimilarLoss(nn.Module):
    """卫星热传导相似性损失函数"""
    
    def __init__(self, lamb=0.2, save_detail_file=None):
        super().__init__()
        # 导入卫星专用模块，复用现有函数
        self.y_norm = DataNormer(data_type="y_norm", config_name='norm_satellite.yaml')
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # 获取卫星输出量纲矩阵，复用现有函数
        _, output_dict = get_bc_dict_satellite()
        self.register_buffer('output_mat', torch.tensor(get_mat_inorder(output_dict), dtype=torch.float32))
        
        # 获取并存储正交基（用于验证，主要逻辑在AugmentSimilar中）
        self.orthogonal_basis = find_orthogonal_basis_satellite()
        print(f"✅ SatelliteSimilarLoss初始化完成，使用{len(self.orthogonal_basis)}个正交基")
        
        self.lamb = lamb
        self.save_detail_file = save_detail_file

    def forward(self, logits, label, logits_n, logits_f, coef):
        """
        卫星相似性损失计算
        
        Args:
            logits: anchor预测 (batch, 256, 256, 1)
            label: 真实标签 (batch, 256, 256, 1)
            logits_n: near预测 (batch, 256, 256, 1)
            logits_f: far预测 (batch, 256, 256, 1)
            coef: 相似变换系数 (batch, 4) - 卫星用4维：[M, L, T, Θ]
        """
        # 验证系数维度（卫星用4维：M,L,T,Θ）
        coef = coef.reshape([-1, 4])
        
        # 主损失：监督学习损失
        loss_0 = self.loss_fn(logits, label)
        
        # 相似性变换：将far预测变换到与near预测相同的物理尺度
        logits_f_back = self.y_norm.back(logits_f)                    # 反标准化
        logits_f_transformed = self.sim_transform(logits_f_back, coef * -1)  # 相似变换
        logits_f_norm = self.y_norm.norm(logits_f_transformed)        # 重新标准化
        
        # 相似性损失：变换后的far预测 vs near预测
        loss_1 = self.loss_fn(logits_f_norm, logits_n.detach())
        
        # 记录详细损失
        if self.save_detail_file is not None:
            with open(self.save_detail_file, 'a') as file:
                file.write(f'{loss_0.item()}\t{loss_1.item()}\n')
        
        # 加权总损失
        return loss_0 * (1 - self.lamb) + loss_1 * self.lamb

    def get_loss_components(self, logits, label, logits_n, logits_f, coef):
        """获取损失组件，用于分析"""
        coef = coef.reshape([-1, 4])
        logits_f_back = self.y_norm.back(logits_f)
        logits_f_transformed = self.sim_transform(logits_f_back, coef * -1)
        logits_f_norm = self.y_norm.norm(logits_f_transformed)
        
        loss_0 = self.loss_fn(logits, label)
        loss_1 = self.loss_fn(logits_f_norm, logits_n)
        
        return loss_0, loss_1

    def sim_transform(self, logits, bc_dim_coef):
        """卫星热传导物理相似变换"""
        from src.data_utils import dimension_scaling_Tensor
        return dimension_scaling_Tensor(
            logits,
            bc_dim_mat=self.output_mat,  # 卫星输出量纲矩阵 (1, 4)
            bc_dim_coef=bc_dim_coef,     # 变换系数 (batch, 4)
        )


