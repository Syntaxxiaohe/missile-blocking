#!/bin/bash
# ============================================================
# M1_NaiveGA 云端运行脚本
# 注意：mode_1 和 mode_2 已在本地完成，此处只跑 mode_3 和 mode_4
# 用法：nohup bash run_M1.sh > log_M1.log 2>&1 &
# ============================================================

set -e
source activate uavsim 2>/dev/null || conda activate uavsim 2>/dev/null || true

N_EVALS=10
N_WORKERS=10
POP_SIZE=30
MAX_GEN=50

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}/M1_NaiveGA"

echo "[$(date '+%H:%M:%S')] >>> M1 mode_3_asymmetric_saturation 开始"
python main.py --mode multi_eval \
    --pattern mode_3_asymmetric_saturation \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M1 mode_3 完成"

echo "[$(date '+%H:%M:%S')] >>> M1 mode_4_full_360_swarm 开始"
python main.py --mode multi_eval \
    --pattern mode_4_full_360_swarm \
    --n_evals ${N_EVALS} --n_workers ${N_WORKERS} \
    --pop_size ${POP_SIZE} --max_gen ${MAX_GEN}
echo "[$(date '+%H:%M:%S')] >>> M1 mode_4 完成"

echo "[$(date '+%H:%M:%S')] ========== M1 全部完成 =========="
