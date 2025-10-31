#!/bin/bash

# ============================================================================
# 批量评估训练结果脚本
# 功能：遍历所有训练结果文件夹，调用eval_satellite_predict.py生成评估结果和npy文件
# ============================================================================

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "项目根目录: $PROJECT_ROOT"
echo "============================================================================"

# CUDA设备设置
export CUDA_VISIBLE_DEVICES=6

# 日志文件
LOG_FILE="batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "开始批量评估: $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

# 创建输出根目录
OUTPUT_ROOT="work_post_eval"
mkdir -p "$OUTPUT_ROOT"
echo "输出目录: $OUTPUT_ROOT" | tee -a "$LOG_FILE"

# 主文件夹列表
WORK_DIRS=(
    # "work_satellite_BASE_fno_20251024"
    # "work_satellite_BASE_mlp_20251024"
    # "work_satellite_dssl_fno_20251021"
    "work_satellite_dssl_mlp_20251021"
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
    
    # 创建对应的输出子目录
    OUTPUT_SUBDIR="$OUTPUT_ROOT/$work_dir"
    mkdir -p "$OUTPUT_SUBDIR"
    
    # 遍历子文件夹
    for ckpt_dir in "$work_dir"/*; do
        
        # 检查是否为目录
        if [ ! -d "$ckpt_dir" ]; then
            continue
        fi
        
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        
        # 检查是否包含必要文件
        if [ ! -f "$ckpt_dir/latest_model.pth" ]; then
            echo "[跳过] 缺少latest_model.pth: $ckpt_dir" | tee -a "$LOG_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        # 提取文件夹名称
        folder_name=$(basename "$ckpt_dir")
        
        # 提取模型类型（FNO或MLP，转小写）
        model_type=$(echo "$folder_name" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]')
        
        # 提取ntrain（匹配 _n数字_ 或 _n数字$ 模式）
        ntrain=$(echo "$folder_name" | grep -oP '(?<=_n)\d+(?=_|\b)')
        
        if [ -z "$ntrain" ]; then
            echo "[跳过] 无法提取ntrain: $folder_name" | tee -a "$LOG_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        echo "" | tee -a "$LOG_FILE"
        echo "[开始] 处理: $ckpt_dir" | tee -a "$LOG_FILE"
        echo "  模型类型: $model_type" | tee -a "$LOG_FILE"
        echo "  训练样本数: $ntrain" | tee -a "$LOG_FILE"
        
        # 设置输出目录（保持与原始文件夹相同的层次结构）
        EVAL_OUTPUT_DIR="$OUTPUT_SUBDIR/$folder_name"
        
        # 检查是否已经处理过（避免重复）
        if [ -f "$EVAL_OUTPUT_DIR/metrics.json" ]; then
            echo "[跳过] 已存在评估结果: $EVAL_OUTPUT_DIR" | tee -a "$LOG_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        # 调用评估脚本
        echo "  执行评估..." | tee -a "$LOG_FILE"
        
        # 使用绝对路径
        CKPT_ABS_PATH="$PROJECT_ROOT/$ckpt_dir"
        
        python Demo/satellite_2d_dssl/eval_satellite_predict.py \
            --model "$model_type" \
            --ckpt "$CKPT_ABS_PATH" \
            --ntrain "$ntrain" \
            --nvalid 500 \
            --split all \
            --cuda_index 0 \
            --batch_size 64 \
            --save_dir "$OUTPUT_ROOT/$work_dir" \
            --max_plots 10 \
            2>&1 | tee -a "$LOG_FILE"
        
        # 检查退出状态
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            # 查找最新生成的eval目录（包含folder_name的前缀）
            LATEST_EVAL=$(ls -td "$OUTPUT_ROOT/$work_dir"/eval_"${model_type}"_* 2>/dev/null | head -1)
            
            if [ -n "$LATEST_EVAL" ] && [ -d "$LATEST_EVAL" ]; then
                # 重命名为与原始文件夹名称一致（去掉eval_前缀和时间戳）
                mv "$LATEST_EVAL" "$EVAL_OUTPUT_DIR" 2>/dev/null
                echo "[成功] 评估完成: $EVAL_OUTPUT_DIR" | tee -a "$LOG_FILE"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "[警告] 评估脚本执行成功但未找到输出目录" | tee -a "$LOG_FILE"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
        else
            echo "[失败] 评估脚本执行失败" | tee -a "$LOG_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
    done
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "批量评估完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
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

