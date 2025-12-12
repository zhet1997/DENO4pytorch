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
)


def load_satellite_data(
    data_path: str = "./data_post/heat_dataset_new.h5", 
    sample_limit: Optional[int] = None,
    noise_type: Optional[str] = None,
    noise_scale: float = 0.0,
    noise_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从h5py格式文件加载卫星热仿真数据，可选择添加固定噪声
    
    Args:
        data_path (str): h5文件路径
        sample_limit (Optional[int]): 限制加载的样本数量，None表示加载全部数据
        noise_type (Optional[str]): 噪声类型，'independent'(独立噪声) 或 'correlated'(空间相关噪声)，
                                     None 表示不添加噪声
        noise_scale (float): 噪声幅度系数，相对于数据 std 的比例。例如 0.1 表示噪声 std 为数据 std 的 10%
        noise_dir (Optional[str]): 噪声文件所在目录，None 表示与 data_path 同目录
        
    Returns:
        tuple: (输入数据, 输出数据)
               输入：形状为[N, 256, 256, 6] (component_sdf, component_power, cooling_sdf, 
                                         cooling_temp, coord_x, coord_y)
               输出：形状为[N, 256, 256, 1] (temperature)
               
    数据结构：
        
        - inputs通道1: 元件功率密度 (W/m²)
        - inputs通道2: 散热窗SDF 
        - inputs通道3: 散热窗温度 (K)
        - inputs通道4: 坐标X (m) - 以中心为原点
        - inputs通道5: 坐标Y (m) - 以中心为原点
        - inputs通道6: 元件SDF (signed distance field)
        
        
        - outputs通道0: 温度场 (K)
        
    噪声添加说明：
        - 噪声仅添加到前 5000 个训练样本的 output
        - 噪声是预生成的固定值（从 H5 文件加载），确保可复现
        - 实际噪声幅度 = 预生成噪声 × (output_std × noise_scale)
        - 数学：u'(x) = u(x) + α·n(x)，其中 α = output_std × noise_scale
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
            
            # 添加噪声（如果指定）
            if noise_type is not None and noise_scale > 0:
                print(f"\n🔊 添加固定噪声:")
                print(f"   - 噪声类型: {noise_type}")
                print(f"   - 噪声系数: {noise_scale}")
                
                # 确定噪声文件目录
                if noise_dir is None:
                    noise_dir = os.path.dirname(data_path)
                
                # 加载噪声文件
                noise_filename = f'noise_{noise_type}.h5'
                noise_path = os.path.join(noise_dir, noise_filename)
                
                if not os.path.exists(noise_path):
                    raise FileNotFoundError(
                        f"噪声文件不存在: {noise_path}\n"
                        f"请先运行 generate_noise.py 生成噪声文件"
                    )
                
                with h5py.File(noise_path, 'r') as nf:
                    noise_data = np.array(nf['noise'], dtype=np.float32)
                
                print(f"   - 噪声文件: {noise_path}")
                print(f"   - 噪声形状: {noise_data.shape}")
                
                # 检查噪声形状是否匹配
                n_train = min(5000, outputs.shape[0])
                expected_noise_shape = (n_train, 256, 256, 1)
                if noise_data.shape[0] < n_train:
                    raise ValueError(
                        f"噪声样本数量不足: {noise_data.shape[0]} < {n_train}\n"
                        f"请重新生成噪声文件，确保至少包含 {n_train} 个样本"
                    )
                if noise_data.shape[1:] != (256, 256, 1):
                    raise ValueError(
                        f"噪声空间维度不匹配: {noise_data.shape[1:]} != (256, 256, 1)\n"
                        f"期望形状: {expected_noise_shape}"
                    )
                
                # 计算前 5000 个样本的 output 标准差
                output_std = np.std(outputs[:n_train])
                print(f"   - 训练集 output std: {output_std:.4f}")
                
                # 计算噪声数据本身的 std（用于归一化）
                noise_std = np.std(noise_data[:n_train])
                print(f"   - 噪声数据 std: {noise_std:.4f}")
                
                # 修正：先归一化噪声到 std=1，再按目标 std 缩放
                # 公式: actual_noise = (noise_data / noise_data.std()) * (output_std * noise_scale)
                # 这样确保 noise_scale=1.0 时，实际噪声 std = output_std
                actual_noise = (noise_data[:n_train] / noise_std) * (output_std * noise_scale)
                actual_noise_std = np.std(actual_noise)
                print(f"   - 实际噪声 std: {actual_noise_std:.4f} (期望: {output_std * noise_scale:.4f})")
                print(f"   - 噪声强度: {noise_scale*100:.1f}% of data std")
                
                # 只对前 5000 个样本添加噪声
                outputs[:n_train] = outputs[:n_train] + actual_noise
                print(f"   - 噪声已添加到前 {n_train} 个样本")
            
            return inputs, outputs
            
    except Exception as e:
        raise RuntimeError(f"加载卫星数据失败: {e}")


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


