#!/bin/bash

# ============================================================================
# 批量绘制回归误差图脚本
# 功能：遍历 work_post_eval/ 目录下所有评估结果，为每个目录生成两张回归图
#       - regression_mean.png (平均温度)
#       - regression_max.png (最高温度)
# ============================================================================

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "项目根目录: $PROJECT_ROOT"
echo "============================================================================"

# 日志文件
LOG_FILE="batch_plot_regression_$(date +%Y%m%d_%H%M%S).log"
echo "开始批量绘制回归图: $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

# 工作目录
WORK_DIR="work_post_eval"

# 检查工作目录是否存在
if [ ! -d "$WORK_DIR" ]; then
    echo "[错误] 目录不存在: $WORK_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# 统计变量
TOTAL_DIRS=0
SUCCESS_DIRS=0
FAIL_DIRS=0
SKIP_DIRS=0

# 开始时间
START_TIME=$(date +%s)

# 遍历主文件夹
for main_dir in "$WORK_DIR"/*; do
    
    # 检查是否为目录
    if [ ! -d "$main_dir" ]; then
        continue
    fi
    
    main_dir_name=$(basename "$main_dir")
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "处理主文件夹: $main_dir_name" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 遍历子文件夹
    for sub_dir in "$main_dir"/*; do
        
        # 检查是否为目录
        if [ ! -d "$sub_dir" ]; then
            continue
        fi
        
        TOTAL_DIRS=$((TOTAL_DIRS + 1))
        sub_dir_name=$(basename "$sub_dir")
        
        # 检查是否包含必要的npy文件
        if [ ! -f "$sub_dir/train_pred.npy" ] || \
           [ ! -f "$sub_dir/train_true.npy" ] || \
           [ ! -f "$sub_dir/valid_pred.npy" ] || \
           [ ! -f "$sub_dir/valid_true.npy" ]; then
            echo "[跳过] 缺少npy文件: $sub_dir_name" | tee -a "$LOG_FILE"
            SKIP_DIRS=$((SKIP_DIRS + 1))
            continue
        fi
        
        echo "" | tee -a "$LOG_FILE"
        echo "[开始] 处理: $sub_dir_name" | tee -a "$LOG_FILE"
        
        # 标记是否成功
        ALL_SUCCESS=true
        
        # 1. 绘制 mean 方法的回归图
        echo "  [mean] 绘制平均温度回归图..." | tee -a "$LOG_FILE"
        python Demo/satellite_2d_dssl/plot_regression_from_npy.py \
            --eval_dir "$sub_dir" \
            --method mean \
            --output_name regression_mean.png \
            2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  [mean] 成功: $sub_dir/regression_mean.png" | tee -a "$LOG_FILE"
        else
            echo "  [mean] 失败" | tee -a "$LOG_FILE"
            ALL_SUCCESS=false
        fi
        
        # 2. 绘制 max 方法的回归图
        echo "  [max] 绘制最高温度回归图..." | tee -a "$LOG_FILE"
        python Demo/satellite_2d_dssl/plot_regression_from_npy.py \
            --eval_dir "$sub_dir" \
            --method max \
            --output_name regression_max.png \
            2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  [max] 成功: $sub_dir/regression_max.png" | tee -a "$LOG_FILE"
        else
            echo "  [max] 失败" | tee -a "$LOG_FILE"
            ALL_SUCCESS=false
        fi
        
        # 统计结果
        if [ "$ALL_SUCCESS" = true ]; then
            echo "[成功] 完成: $sub_dir_name" | tee -a "$LOG_FILE"
            SUCCESS_DIRS=$((SUCCESS_DIRS + 1))
        else
            echo "[失败] 部分失败: $sub_dir_name" | tee -a "$LOG_FILE"
            FAIL_DIRS=$((FAIL_DIRS + 1))
        fi
        
    done
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "批量绘制完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "统计信息:" | tee -a "$LOG_FILE"
echo "  总目录数: $TOTAL_DIRS" | tee -a "$LOG_FILE"
echo "  成功: $SUCCESS_DIRS" | tee -a "$LOG_FILE"
echo "  失败: $FAIL_DIRS" | tee -a "$LOG_FILE"
echo "  跳过: $SKIP_DIRS" | tee -a "$LOG_FILE"
echo "  总耗时: ${ELAPSED}秒 ($(($ELAPSED / 60))分钟)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "输出说明:" | tee -a "$LOG_FILE"
echo "  每个评估目录包含两张回归图:" | tee -a "$LOG_FILE"
echo "    - regression_mean.png (平均温度回归，Train和Valid的R²分别计算)" | tee -a "$LOG_FILE"
echo "    - regression_max.png  (最高温度回归，Train和Valid的R²分别计算)" | tee -a "$LOG_FILE"

