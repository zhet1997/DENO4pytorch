#!/bin/bash

# 噪声训练实验脚本
# 测试不同样本数、噪声强度和噪声类型
# 并行使用多张显卡 (4, 6, 7)，每张卡同时运行一个任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Python 脚本目录（相对于 shell 脚本）
RUN_SCRIPT_DIR="$(cd "$SCRIPT_DIR/../run_script" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 配置参数
CUDA_DEVICES=(5 6 7)
NTRAIN_LIST=(2000 4000)
NOISE_STD_LIST=(0.005 0.01 0.05 0.1 0.2)
NOISE_TYPE_LIST=(independent correlated)  # 噪声类型：独立噪声 或 空间相关噪声
NVALID=500
SELFSUP_DIR="/data/wqn/datasets/packaged_dataset20251017_6c_sim1/"
MAX_PARALLEL=${#CUDA_DEVICES[@]}  # 最大并行任务数 = 显卡数

LOG_FILE="experiments_noise_$(date +%Y%m%d_%H%M%S).log"
echo "噪声实验开始: $(date)" | tee -a "$LOG_FILE"
echo "并行显卡数: $MAX_PARALLEL (${CUDA_DEVICES[*]})" | tee -a "$LOG_FILE"
echo "Python 脚本目录: $RUN_SCRIPT_DIR" | tee -a "$LOG_FILE"

# 任务队列和显卡分配
declare -a task_queue=()
declare -a gpu_queue=()

# 构建任务队列
for ntrain in "${NTRAIN_LIST[@]}"; do
    for noise in "${NOISE_STD_LIST[@]}"; do
        for noise_type in "${NOISE_TYPE_LIST[@]}"; do
            for script in "run_MLP_satellite_base_noise.py" \
                          "run_MLP_satellite_selfsup_noise.py" \
                          "run_FNO_satellite_base_noise.py" \
                          "run_FNO_satellite_selfsup_noise.py"; do
                task_queue+=("$ntrain|$noise|$noise_type|$script")
            done
        done
    done
done

echo "总任务数: ${#task_queue[@]}" | tee -a "$LOG_FILE"

# 并行执行函数
run_task() {
    local ntrain=$1
    local noise=$2
    local noise_type=$3
    local script=$4
    local cuda=$5
    
    # 确定工作目录和脚本类型
    if [[ $script == *"MLP"* ]]; then
        model="MLP"
    else
        model="FNO"
    fi
    
    if [[ $script == *"selfsup"* ]]; then
        type="DSSL"
        work_dir="work_satellite_noise_${model,,}_dssl"
    else
        type="BASE"
        work_dir="work_satellite_noise_${model,,}_base"
    fi
    
    echo "[${model}_${type}] n=$ntrain, noise=$noise, type=$noise_type, CUDA=$cuda 开始: $(date)" | tee -a "$LOG_FILE"
    
    # 执行训练，输出重定向到单独的日志
    task_log="${LOG_FILE%.log}_${model}_${type}_n${ntrain}_noise${noise}_${noise_type}.log"
    script_path="$RUN_SCRIPT_DIR/$script"
    
    CUDA_VISIBLE_DEVICES=$cuda python "$script_path" \
        --ntrain $ntrain \
        --nvalid $NVALID \
        --noise_std $noise \
        --noise_type $noise_type \
        --cuda_index 0 \
        --work_dir $work_dir \
        --selfsup_dir "$SELFSUP_DIR" > "$task_log" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[${model}_${type}] n=$ntrain, noise=$noise, type=$noise_type 完成 ✓ $(date)" | tee -a "$LOG_FILE"
    else
        echo "[${model}_${type}] n=$ntrain, noise=$noise, type=$noise_type 失败 ✗ $(date)" | tee -a "$LOG_FILE"
    fi
    
    return $status
}

# 主循环：管理并行任务
task_idx=0
gpu_idx=0
declare -A running_pids  # 关联数组：PID -> GPU_ID

while [ $task_idx -lt ${#task_queue[@]} ] || [ ${#running_pids[@]} -gt 0 ]; do
    # 启动新任务（如果有空闲显卡且有待执行任务）
    while [ ${#running_pids[@]} -lt $MAX_PARALLEL ] && [ $task_idx -lt ${#task_queue[@]} ]; do
        task_info="${task_queue[$task_idx]}"
        # 解析: ntrain|noise|noise_type|script
        ntrain="${task_info%%|*}"
        temp="${task_info#*|}"
        noise="${temp%%|*}"
        temp="${temp#*|}"
        noise_type="${temp%%|*}"
        script="${temp##*|}"
        cuda="${CUDA_DEVICES[$gpu_idx]}"
        
        # 后台执行任务
        run_task "$ntrain" "$noise" "$noise_type" "$script" "$cuda" &
        pid=$!
        running_pids[$pid]=$cuda
        
        task_idx=$((task_idx + 1))
        gpu_idx=$(( (gpu_idx + 1) % ${#CUDA_DEVICES[@]} ))
    done
    
    # 检查并清理完成的任务
    for pid in "${!running_pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            wait $pid  # 获取退出状态
            unset running_pids[$pid]
        fi
    done
    
    # 短暂休眠避免CPU占用过高
    sleep 2
done

echo "========================================" | tee -a "$LOG_FILE"
echo "所有实验完成: $(date)" | tee -a "$LOG_FILE"

