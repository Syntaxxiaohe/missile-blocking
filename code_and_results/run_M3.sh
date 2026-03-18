#!/bin/bash
# ============================================================
# M3_AmmoPenalty 云端运行脚本（全部 4 个模式）
# 用法：nohup bash run_M3.sh > log_M3.log 2>&1 &
# ============================================================

set -e
source activate uavsim 2>/dev/null || conda activate uavsim 2>/dev/null || true

N_EVALS=10
N_WORKERS=10
POP_SIZE=30
MAX_GEN=50

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}/M3_AmmoPenalty"

echo "[$(date '+%H:%M:%S')] >>> M3 mode_1_single_sector 开始"
python main.py --mode multi_eval \
    --pattern mode_1_single_sector \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M3 mode_1 完成"

echo "[$(date '+%H:%M:%S')] >>> M3 mode_2_orthogonal_pincer 开始"
python main.py --mode multi_eval \
    --pattern mode_2_orthogonal_pincer \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M3 mode_2 完成"

echo "[$(date '+%H:%M:%S')] >>> M3 mode_3_asymmetric_saturation 开始"
python main.py --mode multi_eval \
    --pattern mode_3_asymmetric_saturation \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M3 mode_3 完成"

echo "[$(date '+%H:%M:%S')] >>> M3 mode_4_full_360_swarm 开始"
python main.py --mode multi_eval \
    --pattern mode_4_full_360_swarm \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M3 mode_4 完成"

echo "[$(date '+%H:%M:%S')] ========== M3 全部完成 =========="
