"""
诊断噪声加载问题

分析噪声归一化方式对实际噪声强度的影响
"""
import os
import h5py
import numpy as np

def diagnose_noise_issue():
    """诊断噪声加载问题"""
    print("=" * 60)
    print("噪声加载问题诊断")
    print("=" * 60)
    
    # 1. 检查噪声文件
    noise_dir = '/data/wqn/datasets/packaged_dataset20251017_6c'
    noise_path = os.path.join(noise_dir, 'noise_independent.h5')
    
    if not os.path.exists(noise_path):
        print(f"❌ 噪声文件不存在: {noise_path}")
        return
    
    # 2. 加载噪声数据
    with h5py.File(noise_path, 'r') as f:
        noise_data = np.array(f['noise'], dtype=np.float32)
    
    print(f"\n📊 噪声数据统计:")
    print(f"   - 形状: {noise_data.shape}")
    print(f"   - 范围: [{noise_data.min():.4f}, {noise_data.max():.4f}]")
    print(f"   - 均值: {noise_data.mean():.6f}")
    print(f"   - 标准差: {noise_data.std():.4f}")
    print(f"   - 最大值绝对值: {np.max(np.abs(noise_data)):.4f}")
    
    # 3. 加载输出数据
    data_path = os.path.join(noise_dir, 'heat_dataset.h5')
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        outputs = np.array(f['outputs'], dtype=np.float32)
    
    n_train = min(5000, outputs.shape[0])
    output_std = np.std(outputs[:n_train])
    
    print(f"\n📊 输出数据统计（前{n_train}个样本）:")
    print(f"   - 形状: {outputs[:n_train].shape}")
    print(f"   - 标准差: {output_std:.4f}")
    
    # 4. 模拟当前噪声添加方式
    print(f"\n🔍 当前噪声添加方式分析:")
    print(f"   公式: actual_noise = noise_data * (output_std * noise_scale)")
    
    noise_scale = 1.0
    actual_noise = noise_data[:n_train] * (output_std * noise_scale)
    actual_noise_std = np.std(actual_noise)
    
    print(f"\n   当 noise_scale = {noise_scale} 时:")
    print(f"   - 期望噪声 std: {output_std:.4f} (应该等于 output_std)")
    print(f"   - 实际噪声 std: {actual_noise_std:.4f}")
    print(f"   - 实际/期望比例: {actual_noise_std / output_std:.4f}")
    
    # 5. 问题分析
    print(f"\n⚠️  问题诊断:")
    noise_std = noise_data.std()
    print(f"   - 噪声数据本身的 std: {noise_std:.4f}")
    print(f"   - 由于噪声被归一化到 [-1, 1]，其 std 很小")
    print(f"   - 实际噪声 std = 噪声数据 std × output_std × noise_scale")
    print(f"   - 实际噪声 std = {noise_std:.4f} × {output_std:.4f} × {noise_scale:.1f} = {actual_noise_std:.4f}")
    print(f"   - 这意味着 noise_scale=1.0 时，实际噪声只有期望的 {actual_noise_std/output_std:.1%}")
    
    # 6. 正确的缩放方式
    print(f"\n✅ 正确的缩放方式应该是:")
    print(f"   公式: actual_noise = (noise_data / noise_data.std()) * (output_std * noise_scale)")
    
    corrected_noise = (noise_data[:n_train] / noise_data.std()) * (output_std * noise_scale)
    corrected_noise_std = np.std(corrected_noise)
    
    print(f"\n   修正后（noise_scale = {noise_scale}）:")
    print(f"   - 期望噪声 std: {output_std:.4f}")
    print(f"   - 实际噪声 std: {corrected_noise_std:.4f}")
    print(f"   - 实际/期望比例: {corrected_noise_std / output_std:.4f}")
    
    # 7. 形状检查
    print(f"\n📐 形状匹配检查:")
    print(f"   - noise_data[:n_train].shape: {noise_data[:n_train].shape}")
    print(f"   - outputs[:n_train].shape: {outputs[:n_train].shape}")
    
    if noise_data[:n_train].shape == outputs[:n_train].shape:
        print(f"   ✅ 形状匹配")
    else:
        print(f"   ❌ 形状不匹配！这会导致广播问题")
        print(f"   - 尝试广播后的形状: {(noise_data[:n_train] + outputs[:n_train]).shape}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    diagnose_noise_issue()

