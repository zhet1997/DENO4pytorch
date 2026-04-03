"""
测试JSON序列化修复
"""
import numpy as np
import json
import os
import sys

sys.path.append('/data/wqn/Code/DENO4pytorch')

from Demo.satellite_sup_2d.distill_modules import RoundLogger

# 创建测试数据（包含各种numpy类型）
test_metrics = {
    'round_id': 0,
    'float32_value': np.float32(0.123456),
    'float64_value': np.float64(0.789012),
    'int32_value': np.int32(42),
    'int64_value': np.int64(100),
    'numpy_array': np.array([1.0, 2.0, 3.0]),
    'm_C': {
        1: np.float32(0.05),
        2: np.float64(0.04),
        3: np.float32(0.03)
    },
    'm_CS': {
        1: np.float32(0.02),
        2: np.float32(0.015),
        3: np.float32(0.01)
    },
    'T': [1, 2, 3],
    'n_star': 3,
    'regular_float': 0.5,
    'regular_int': 10
}

# 测试RoundLogger
logger = RoundLogger('/tmp/test_json.log')
logger.log_round(0, test_metrics)

# 尝试保存
try:
    logger.save_summary('/tmp/test_summary.json')
    print("✓ JSON序列化成功！")
    
    # 读取验证
    with open('/tmp/test_summary.json', 'r') as f:
        loaded_data = json.load(f)
    
    print(f"✓ JSON加载成功！")
    print(f"  记录数量: {len(loaded_data)}")
    print(f"  float32值类型: {type(loaded_data[0]['float32_value'])}")
    print(f"  m_C[1]类型: {type(loaded_data[0]['m_C']['1'])}")
    print(f"  numpy数组类型: {type(loaded_data[0]['numpy_array'])}")
    
    # 清理
    os.remove('/tmp/test_json.log')
    os.remove('/tmp/test_summary.json')
    
    print("\n✓ 所有测试通过！JSON序列化问题已修复。")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()


