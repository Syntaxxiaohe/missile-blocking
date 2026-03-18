#!/bin/bash
# ============================================================
# 云端环境部署脚本（一次性运行）
# 用法：bash deploy_cloud.sh
# ============================================================

set -e  # 任何步骤报错立即停止

echo "=============================="
echo "  Step 1: 创建 conda 环境"
echo "=============================="
conda create -n uavsim python=3.10 -y
source activate uavsim || conda activate uavsim

echo "=============================="
echo "  Step 2: 安装依赖"
echo "=============================="
pip install -r requirements.txt

echo "=============================="
echo "  Step 3: 创建结果目录"
echo "=============================="
mkdir -p M1_NaiveGA/results
mkdir -p M2_Clustering/results
mkdir -p M3_AmmoPenalty/results

echo "=============================="
echo "  Step 4: 验证环境（快速测试）"
echo "=============================="
python -c "import numpy; print('numpy OK:', numpy.__version__)"
python -c "import multiprocessing; print('CPU 核心数:', multiprocessing.cpu_count())"

echo ""
echo "=============================="
echo "  部署完成！请运行 bash run_cloud.sh 开始实验"
echo "=============================="
