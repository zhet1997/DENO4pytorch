#!/bin/bash

# 循环式叠加蒸馏框架 - 快速启动脚本
# 用途：快速测试完整训练流程（小规模参数）

echo "============================================================"
echo "循环式叠加蒸馏框架 - 快速测试"
echo "============================================================"
echo ""

# 设置工作目录
cd /data/wqn/Code/DENO4pytorch

# 测试选项
TEST_TYPE=${1:-"quick"}  # quick | medium | full

case $TEST_TYPE in
    "quick")
        echo "模式: 快速测试（约15分钟）"
        echo "------------------------------------------------------------"
        python Demo/satellite_sup_2d/train_distillation_loop.py \
            --max_rounds 2 \
            --supervised_epochs 10 \
            --consistency_epochs 3 \
            --distill_epochs 5 \
            --batch_size 8 \
            --samples_per_n 100 \
            --component_nums 1 2 3 \
            --anchor_samples 20 \
            --tau_rmse 0.08 \
            --delta_rmse 0.005 \
            --lambda_distill 0.5 \
            --patience 2 \
            --work_name quick_test
        ;;
    
    "medium")
        echo "模式: 中等规模测试（约1小时）"
        echo "------------------------------------------------------------"
        python Demo/satellite_sup_2d/train_distillation_loop.py \
            --max_rounds 5 \
            --supervised_epochs 30 \
            --consistency_epochs 10 \
            --distill_epochs 15 \
            --batch_size 16 \
            --samples_per_n 500 \
            --component_nums 1 2 3 4 5 6 7 \
            --anchor_samples 50 \
            --tau_rmse 0.05 \
            --delta_rmse 0.01 \
            --lambda_distill 0.5 \
            --patience 3 \
            --work_name medium_test
        ;;
    
    "full")
        echo "模式: 完整训练（约3-5小时）"
        echo "------------------------------------------------------------"
        python Demo/satellite_sup_2d/train_distillation_loop.py \
            --max_rounds 10 \
            --supervised_epochs 100 \
            --consistency_epochs 20 \
            --distill_epochs 30 \
            --batch_size 16 \
            --samples_per_n 1000 \
            --component_nums 1 2 3 4 5 6 7 8 9 10 \
            --anchor_samples 50 \
            --tau_rmse 0.05 \
            --delta_rmse 0.01 \
            --lambda_distill 0.5 \
            --patience 3 \
            --work_name full_training
        ;;
    
    "consistency_only")
        echo "模式: 仅测试一致性训练（约30分钟）"
        echo "------------------------------------------------------------"
        python Demo/satellite_sup_2d/train_consistency.py \
            --epochs 100 \
            --batch_size 16 \
            --lr 1e-3 \
            --consistency_epochs 5 \
            --K 8 \
            --work_name test_consistency
        ;;
    
    *)
        echo "错误: 未知的测试类型 '$TEST_TYPE'"
        echo ""
        echo "用法: bash quick_start.sh [TEST_TYPE]"
        echo ""
        echo "TEST_TYPE 选项:"
        echo "  quick             - 快速测试（约15分钟，默认）"
        echo "  medium            - 中等规模（约1小时）"
        echo "  full              - 完整训练（约3-5小时）"
        echo "  consistency_only  - 仅一致性训练（约30分钟）"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "训练完成！"
echo "============================================================"
echo ""
echo "查看结果:"
echo "  - 日志: work_satellite/${TEST_TYPE}_test/distillation_loop.log"
echo "  - 摘要: work_satellite/${TEST_TYPE}_test/training_summary.json"
echo "  - 模型: work_satellite/${TEST_TYPE}_test/final_model.pth"
echo ""


