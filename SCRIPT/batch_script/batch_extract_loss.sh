#!/bin/bash

# ============================================================================
# 批量提取训练loss统计信息
# 功能：遍历所有训练结果文件夹，提取loss_history.npy的统计信息并保存为JSON
# ============================================================================

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从 SCRIPT/batch_script/ 向上两级到项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "项目根目录: $PROJECT_ROOT"
echo "============================================================================"

# 日志文件
LOG_FILE="batch_extract_loss_$(date +%Y%m%d_%H%M%S).log"
echo "开始批量提取loss统计信息: $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

# 创建输出根目录
OUTPUT_ROOT="work_post_eval"
mkdir -p "$OUTPUT_ROOT"
echo "输出目录: $OUTPUT_ROOT" | tee -a "$LOG_FILE"

# 主文件夹列表（实际路径在 SCRIPT/batch_script/ 下）
WORK_DIRS=(
    "SCRIPT/run_script/work_satellite_dssl_fno_lossgap"
    "SCRIPT/run_script/work_satellite_dssl_mlp_lossgap"
    "SCRIPT/run_script/work_satellite_dssl_trans"
)

# 统计变量
TOTAL_COUNT=0
SUCCESS_COUNT=0

FAIL_COUNT=0
SKIP_COUNT=0

# 开始时间
START_TIME=$(date +%s)

# 遍历每个主文件夹
for work_dir in "${WORK_DIRS[@]}"; do
    
    # 检查主文件夹是否存在
    if [ ! -d "$work_dir" ]; then
        echo "[跳过] 主文件夹不存在: $work_dir" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "处理主文件夹: $work_dir" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 创建对应的输出子目录（使用basename去掉相对路径前缀）
    work_dir_basename=$(basename "$work_dir")
    OUTPUT_SUBDIR="$OUTPUT_ROOT/$work_dir_basename"
    mkdir -p "$OUTPUT_SUBDIR"
    
    # 遍历子文件夹
    for train_dir in "$work_dir"/*; do
        
        # 检查是否为目录
        if [ ! -d "$train_dir" ]; then
            continue
        fi
        
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        
        # 检查是否包含loss_history.npy文件
        if [ ! -f "$train_dir/loss_history.npy" ]; then
            echo "[跳过] 缺少loss_history.npy: $train_dir" | tee -a "$LOG_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        # 提取文件夹名称
        folder_name=$(basename "$train_dir")
        
        # 设置eval输出目录
        EVAL_OUTPUT_DIR="$OUTPUT_SUBDIR/$folder_name"
        
        # 检查是否已经处理过（避免重复）
        if [ -f "$EVAL_OUTPUT_DIR/loss_summary.json" ]; then
            echo "[跳过] 已存在loss_summary.json: $EVAL_OUTPUT_DIR" | tee -a "$LOG_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        echo "" | tee -a "$LOG_FILE"
        echo "[开始] 处理: $train_dir" | tee -a "$LOG_FILE"
        
        # 使用绝对路径
        TRAIN_ABS_PATH="$(cd "$train_dir" && pwd)"
        
        # 调用Python脚本提取loss统计信息
        echo "  执行提取..." | tee -a "$LOG_FILE"
        
        python SCRIPT/plot_script/extract_loss_summary.py \
            --train_dir "$TRAIN_ABS_PATH" \
            --eval_dir "$EVAL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
        
        # 检查退出状态
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            # 验证文件是否生成
            if [ -f "$EVAL_OUTPUT_DIR/loss_summary.json" ]; then
                echo "[成功] loss统计信息已提取: $EVAL_OUTPUT_DIR/loss_summary.json" | tee -a "$LOG_FILE"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "[警告] 脚本执行成功但未找到输出文件" | tee -a "$LOG_FILE"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            echo "[失败] 提取脚本执行失败" | tee -a "$LOG_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
    done
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "批量提取完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "统计信息:" | tee -a "$LOG_FILE"
echo "  总任务数: $TOTAL_COUNT" | tee -a "$LOG_FILE"
echo "  成功: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "  失败: $FAIL_COUNT" | tee -a "$LOG_FILE"
echo "  跳过: $SKIP_COUNT" | tee -a "$LOG_FILE"
echo "  总耗时: ${ELAPSED}秒 ($(($ELAPSED / 60))分钟)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_ROOT" | tee -a "$LOG_FILE"

