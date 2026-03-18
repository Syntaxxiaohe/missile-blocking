#!/bin/bash
# ============================================================
# 云端实验运行脚本
#
# 当前进度（本地已跑完，无需重跑）：
#   M1 mode_1_single_sector      -> results/M1_mode_1_single_sector_20260310_081605.json
#   M1 mode_2_orthogonal_pincer  -> results/M1_mode_2_orthogonal_pincer_20260310_153615.json
#
# 云端需要跑的任务（按顺序执行）：
#   M1: mode_3, mode_4
#   M2: mode_1, mode_2, mode_3, mode_4
#   M3: mode_1, mode_2, mode_3, mode_4
#
# 用法：bash run_cloud.sh
# ============================================================

set -e
source activate uavsim || conda activate uavsim

# ============ 参数配置（保持与本地一致，确保公平对比） ============
N_EVALS=10
N_WORKERS=10
POP_SIZE=30
MAX_GEN=50

echo ""
echo "============================================================"
echo "  实验配置: n_evals=${N_EVALS}  n_workers=${N_WORKERS}"
echo "            pop_size=${POP_SIZE}  max_gen=${MAX_GEN}"
echo "  注意: M1 mode_1/mode_2 已在本地跑完，此处跳过"
echo "============================================================"
echo ""

# ============================================================
# M1 NaiveGA
# ============================================================
echo ">>> [M1] mode_3_asymmetric_saturation 开始..."
cd M1_NaiveGA
python main.py --mode multi_eval \
    --pattern mode_3_asymmetric_saturation \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M1] mode_3 完成"

echo ">>> [M1] mode_4_full_360_swarm 开始..."
python main.py --mode multi_eval \
    --pattern mode_4_full_360_swarm \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M1] mode_4 完成"
cd ..

# ============================================================
# M2 Clustering
# ============================================================
echo ">>> [M2] mode_1_single_sector 开始..."
cd M2_Clustering
python main.py --mode multi_eval \
    --pattern mode_1_single_sector \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M2] mode_1 完成"

echo ">>> [M2] mode_2_orthogonal_pincer 开始..."
python main.py --mode multi_eval \
    --pattern mode_2_orthogonal_pincer \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M2] mode_2 完成"

echo ">>> [M2] mode_3_asymmetric_saturation 开始..."
python main.py --mode multi_eval \
    --pattern mode_3_asymmetric_saturation \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M2] mode_3 完成"

echo ">>> [M2] mode_4_full_360_swarm 开始..."
python main.py --mode multi_eval \
    --pattern mode_4_full_360_swarm \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M2] mode_4 完成"
cd ..

# ============================================================
# M3 AmmoPenalty
# ============================================================
echo ">>> [M3] mode_1_single_sector 开始..."
cd M3_AmmoPenalty
python main.py --mode multi_eval \
    --pattern mode_1_single_sector \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M3] mode_1 完成"

echo ">>> [M3] mode_2_orthogonal_pincer 开始..."
python main.py --mode multi_eval \
    --pattern mode_2_orthogonal_pincer \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M3] mode_2 完成"

echo ">>> [M3] mode_3_asymmetric_saturation 开始..."
python main.py --mode multi_eval \
    --pattern mode_3_asymmetric_saturation \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M3] mode_3 完成"

echo ">>> [M3] mode_4_full_360_swarm 开始..."
python main.py --mode multi_eval \
    --pattern mode_4_full_360_swarm \
    --n_evals ${N_EVALS} \
    --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} \
    --max_gen ${MAX_GEN}
echo ">>> [M3] mode_4 完成"
cd ..

echo ""
echo "============================================================"
echo "  全部实验完成！结果文件位于："
echo "    M1_NaiveGA/results/"
echo "    M2_Clustering/results/"
echo "    M3_AmmoPenalty/results/"
echo "============================================================"
