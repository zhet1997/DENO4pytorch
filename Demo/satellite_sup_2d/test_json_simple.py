"""
简单的JSON序列化测试（不依赖其他模块）
"""
import numpy as np
import json

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

# 测试数据（模拟训练指标）
test_data = {
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
    'nested_dict': {
        'level2': {
            'float32': np.float32(1.23),
            'array': np.array([4, 5, 6])
        }
    }
}

print("原始数据类型:")
print(f"  float32_value: {type(test_data['float32_value'])}")
print(f"  m_C[1]: {type(test_data['m_C'][1])}")
print(f"  numpy_array: {type(test_data['numpy_array'])}")

# 转换
converted = convert_to_json_serializable(test_data)

print("\n转换后类型:")
print(f"  float32_value: {type(converted['float32_value'])}")
print(f"  m_C[1]: {type(converted['m_C'][1])}")
print(f"  numpy_array: {type(converted['numpy_array'])}")

# 尝试JSON序列化
try:
    json_str = json.dumps(converted, indent=2)
    print("\n✓ JSON序列化成功！")
    
    # 尝试反序列化
    loaded = json.loads(json_str)
    print("✓ JSON反序列化成功！")
    
    # 验证值
    print(f"\n值验证:")
    print(f"  float32_value: {loaded['float32_value']} (原值: {test_data['float32_value']})")
    print(f"  m_C[1]: {loaded['m_C']['1']} (原值: {test_data['m_C'][1]})")
    print(f"  numpy_array: {loaded['numpy_array']} (原值: {test_data['numpy_array'].tolist()})")
    
    print("\n✓✓✓ 所有测试通过！JSON序列化问题已修复。")
    
except TypeError as e:
    print(f"\n✗ JSON序列化失败: {e}")
    import traceback
    traceback.print_exc()


