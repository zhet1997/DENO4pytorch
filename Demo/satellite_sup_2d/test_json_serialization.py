"""
测试JSON序列化修复
"""

import numpy as np
import json
import os
import sys
sys.path.append('/data/wqn/Code/DENO4pytorch')

from Demo.satellite_sup_2d.distill_modules import RoundLogger


def test_json_serialization():
    """测试包含numpy类型的metrics是否能正确序列化"""
    
    print("="*60)
    print("测试JSON序列化（numpy类型）")
    print("="*60)
    
    # 创建临时日志路径
    test_log_path = '/tmp/test_round.log'
    test_summary_path = '/tmp/test_summary.json'
    
    # 创建RoundLogger
    logger = RoundLogger(test_log_path)
    
    # 模拟包含numpy类型的metrics（这是导致错误的原因）
    test_metrics_round_0 = {
        'supervised_train_loss': np.float32(0.0123),  # numpy float32
        'supervised_valid_loss': np.float64(0.0156),  # numpy float64
        'm_C': {
            1: np.float32(0.08),
            2: np.float32(0.06),
            3: np.float32(0.04)
        },
        'm_CS': {
            1: np.float64(0.03),
            2: np.float64(0.025),
            3: np.float64(0.02)
        },
        'T': [1, 2, 3],
        'n_star': np.int64(3),  # numpy int
        'interval_A': 1,
        'interval_B': 3,
    }
    
    test_metrics_round_1 = {
        'supervised_train_loss': np.float32(0.0098),
        'supervised_valid_loss': np.float64(0.0112),
        'm_C': {
            1: np.float32(0.075),
            2: np.float32(0.055),
            3: np.float32(0.038),
            4: np.float32(0.03)
        },
        'm_CS': {
            1: np.float64(0.028),
            2: np.float64(0.023),
            3: np.float64(0.018),
            4: np.float64(0.015)
        },
        'T': [1, 2, 3, 4],
        'n_star': np.int64(4),
        'interval_A': 2,
        'interval_B': 4,
    }
    
    # 记录多个rounds
    print("\n记录Round 0...")
    logger.log_round(0, test_metrics_round_0)
    
    print("记录Round 1...")
    logger.log_round(1, test_metrics_round_1)
    
    # 尝试保存摘要（这里之前会出错）
    print("\n保存JSON摘要...")
    try:
        logger.save_summary(test_summary_path)
        print("✓ JSON保存成功！")
        
        # 验证能够重新加载
        with open(test_summary_path, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"✓ JSON加载成功！共{len(loaded_data)}个rounds")
        
        # 验证类型转换正确
        print("\n验证数据类型转换:")
        round_0 = loaded_data[0]
        print(f"  supervised_train_loss: {type(round_0['supervised_train_loss'])} = {round_0['supervised_train_loss']}")
        print(f"  n_star: {type(round_0['n_star'])} = {round_0['n_star']}")
        print(f"  m_C[1]: {type(round_0['m_C']['1'])} = {round_0['m_C']['1']}")
        
        # 清理临时文件
        if os.path.exists(test_log_path):
            os.remove(test_log_path)
        if os.path.exists(test_summary_path):
            os.remove(test_summary_path)
        
        print("\n" + "="*60)
        print("测试通过！所有numpy类型已正确转换为Python原生类型")
        print("="*60)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_json_serialization()


