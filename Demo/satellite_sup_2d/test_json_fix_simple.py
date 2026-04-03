"""
简单测试：验证numpy类型JSON序列化修复
"""

import numpy as np
import json


def convert_to_json_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def test_numpy_json_conversion():
    """测试numpy类型转换"""
    
    print("="*60)
    print("测试numpy类型JSON序列化修复")
    print("="*60)
    
    # 模拟原始错误场景：包含numpy类型的数据
    test_data = {
        'round_id': 0,
        'supervised_train_loss': np.float32(0.0123),  # 这会导致JSON序列化失败
        'supervised_valid_loss': np.float64(0.0156),
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
        'n_star': np.int64(3),
        'array_data': np.array([1.0, 2.0, 3.0]),
    }
    
    print("\n原始数据类型:")
    print(f"  supervised_train_loss: {type(test_data['supervised_train_loss'])}")
    print(f"  n_star: {type(test_data['n_star'])}")
    print(f"  m_C[1]: {type(test_data['m_C'][1])}")
    print(f"  array_data: {type(test_data['array_data'])}")
    
    # 测试1：尝试直接序列化（应该失败）
    print("\n测试1：直接JSON序列化（预期失败）...")
    try:
        json.dumps(test_data)
        print("  ✗ 意外成功（应该失败）")
    except TypeError as e:
        print(f"  ✓ 预期的错误: {e}")
    
    # 测试2：使用转换函数后序列化（应该成功）
    print("\n测试2：转换后JSON序列化（预期成功）...")
    try:
        converted_data = convert_to_json_serializable(test_data)
        json_string = json.dumps(converted_data, indent=2)
        print("  ✓ 序列化成功！")
        
        # 验证转换后的类型
        print("\n转换后数据类型:")
        print(f"  supervised_train_loss: {type(converted_data['supervised_train_loss'])}")
        print(f"  n_star: {type(converted_data['n_star'])}")
        print(f"  m_C[1]: {type(converted_data['m_C'][1])}")
        print(f"  array_data: {type(converted_data['array_data'])}")
        
        # 验证能重新加载
        reloaded_data = json.loads(json_string)
        print("\n  ✓ JSON重新加载成功！")
        
        # 验证数值正确性
        print("\n数值验证:")
        print(f"  原始: {test_data['supervised_train_loss']}")
        print(f"  转换: {converted_data['supervised_train_loss']}")
        print(f"  重载: {reloaded_data['supervised_train_loss']}")
        assert abs(float(test_data['supervised_train_loss']) - reloaded_data['supervised_train_loss']) < 1e-6
        print("  ✓ 数值一致！")
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！修复有效")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_numpy_json_conversion()
    exit(0 if success else 1)


