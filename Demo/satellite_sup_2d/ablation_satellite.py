import os
import sys
import yaml
from typing import Tuple, Optional, Dict, Any

# 确保项目根目录在路径中，便于以 `Demo.*` 导入
sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.utilizes_satellite import (
    get_origin_satellite,
    get_loader_satellite,
)


def get_setting_satellite() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    定义卫星数据(17通道)的默认训练/模型配置。
    返回: basic_dict, train_dict, pred_model_dict, super_model_dict
    """
    basic_dict = {
        'in_dim': 17,   # 固定V2格式(17通道)
        'out_dim': 1,
        'ntrain': 1500,
        'nvalid': 300,
    }

    train_dict = {
        'batch_size': 4,  # 减小batch size以适应内存限制
        'epochs': 200,
        'learning_rate': 1e-3,
        'scheduler_step': 100,
        'scheduler_gamma': 0.5,
    }

    # 读取与PakB相同的Transformer配置，设置 node_feats=in_dim
    with open(os.path.join('data', 'configs', 'transformer_config_sate.yml')) as f:
        config = yaml.full_load(f)
        pred_model_dict = config['PakB_2d']
        pred_model_dict['node_feats'] = basic_dict['in_dim']

    super_model_dict = {
        'modes': (16, 16),
        'width': 64,
        'depth': 2,
        'steps': 1,
        'padding': 0,
        'dropout': 0.1,
    }

    return basic_dict, train_dict, pred_model_dict, super_model_dict


def get_loaders_satellite(
    train_num: Optional[int] = None,
    valid_num: Optional[int] = None,
    batch_size: int = 16,
    h5_path: Optional[str] = None,
    shuffled: bool = False,
) -> Tuple[Any, Optional[Any], Any, Any, Dict[str, Any]]:
    """
    构建卫星数据(固定V2/17通道)的训练与验证 DataLoader。
    返回: train_loader, valid_loader, x_normalizer, y_normalizer, meta
    meta: { 'format': 'v2', 'num_components': 12, 'in_dim': 17 }
    """
    train_x, train_y, valid_x, valid_y, fmt, num_components = get_origin_satellite(
        h5_path=h5_path,
        train_num=train_num,
        valid_num=valid_num,
        shuffled=shuffled,
        dataset_format='v2',  # 固定为V2格式
    )

    # 构建 DataLoader（每通道独立归一化）
    train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
        train_x, train_y,
        valid_x, valid_y,
        batch_size=batch_size,
    )

    meta = {
        'format': fmt,
        'num_components': num_components,
        'in_dim': train_x.shape[-1],
    }
    return train_loader, valid_loader, x_norm, y_norm, meta


