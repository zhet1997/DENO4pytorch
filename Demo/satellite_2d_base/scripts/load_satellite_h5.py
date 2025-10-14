import argparse
import os
import sys
import numpy as np

# 允许直接以绝对路径运行脚本时导入项目内模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from Demo.satellite_2d_base.dataset_satellite import load_satellite_data


def print_basic_stats(inputs: np.ndarray, outputs: np.ndarray) -> None:
    print("================ 数据基本信息 ================")
    print(f"样本总数: {inputs.shape[0]}")
    print(f"输入形状: {inputs.shape}  (N, 256, 256, 6)")
    print(f"输出形状: {outputs.shape} (N, 256, 256, 1)")

    input_channel_names = [
        "component_sdf",
        "component_power",
        "cooling_sdf",
        "cooling_temp",
        "coord_x",
        "coord_y",
    ]

    print("\n—— 输入通道统计 ——")
    for ch, name in enumerate(input_channel_names):
        ch_data = inputs[..., ch]
        print(
            f"[{ch}] {name:<16} min={float(np.min(ch_data)):.6f} "
            f"max={float(np.max(ch_data)):.6f} mean={float(np.mean(ch_data)):.6f}"
        )

    print("\n—— 输出通道统计 ——")
    ch_data = outputs[..., 0]
    print(
        f"[0] temperature       min={float(np.min(ch_data)):.6f} "
        f"max={float(np.max(ch_data)):.6f} mean={float(np.mean(ch_data)):.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="加载并校验卫星热数据集（不进行归一化/切分/增强）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/wqn/turbine_uq/data_post/heat_dataset_780.h5",
        help="h5 数据文件绝对路径",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="可选：限制加载的样本数量（调试用）",
    )
    args = parser.parse_args()

    inputs, outputs = load_satellite_data(args.data_path, sample_limit=args.sample_limit)

    # 形状断言
    assert inputs.ndim == 4 and inputs.shape[1:] == (256, 256, 6), (
        f"输入形状错误: {inputs.shape}"
    )
    assert outputs.ndim == 4 and outputs.shape[1:] == (256, 256, 1), (
        f"输出形状错误: {outputs.shape}"
    )

    print_basic_stats(inputs, outputs)
    print("\n✅ 数据加载校验完成。")


if __name__ == "__main__":
    main()


