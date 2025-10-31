#!/bin/bash
# 批量绘制误差分布图脚本
# 功能：遍历 work_post_eval 下所有训练结果目录，为每个目录生成误差分布图

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../Demo/satellite_2d_dssl/plot_error_hist_from_npy.py"

echo "============================================"
echo "批量绘制误差分布图"
echo "============================================"
echo ""

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到脚本 $PYTHON_SCRIPT"
    exit 1
fi

# 查找所有包含 train_true.npy 的目录
eval_dirs=$(find "$SCRIPT_DIR" -name "train_true.npy" -type f | xargs -I {} dirname {})

if [ -z "$eval_dirs" ]; then
    echo "错误: 未找到任何包含 train_true.npy 的目录"
    exit 1
fi

# 统计目录数量
total_dirs=$(echo "$eval_dirs" | wc -l)
echo "找到 $total_dirs 个评估目录"
echo ""

# 遍历每个目录
current=0
for eval_dir in $eval_dirs; do
    current=$((current + 1))
    echo "[$current/$total_dirs] 处理: $eval_dir"
    
    # 绘制误差分布图（逐像素统计模式）
    python "$PYTHON_SCRIPT" \
        --eval_dir "$eval_dir" \
        --error_type mse \
        --plot_mode valid \
        --pixel_wise \
        --output_name error_distribution_valid_pixel_wise.png
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 成功生成图片"
    else
        echo "  ✗ 处理失败"
    fi
    echo ""
done

echo "============================================"
echo "批量处理完成！共处理 $total_dirs 个目录"
echo "============================================"

