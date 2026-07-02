#!/bin/bash
#SBATCH --job-name=nep_bs1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=8375
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 遇到命令错误时立即退出
set -e

# 防止部分算子栈空间不足
ulimit -s unlimited
ulimit -l unlimited

# 清理旧模块
module purge

# 激活 Conda 环境
source /data/home/liuhanyu/miniconda3/etc/profile.d/conda.sh
conda activate ai2pot_env

# CPU or GPU
export CUDA_VISIBLE_DEVICES=""

# CPU 线程数与 --cpus-per-task 保持一致
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURM_JOB_NODELIST}"
echo "CPU threads  : ${SLURM_CPUS_PER_TASK}"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES}"
echo "Python       : $(which python)"
echo "========================================"

# 检查 PyTorch 是否识别 GPU
python - << 'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

try:
    cuda_available = torch.cuda.is_available()
except Exception:
    cuda_available = False

print("CUDA available:", cuda_available)

if cuda_available:
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0

    print("GPU count:", n)

    if n > 0:
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("GPU name: error ->", e)
    else:
        print("GPU name: None (no visible devices)")
else:
    print("GPU count: 0")
    print("GPU name: CPU-only environment")
PY

# 启动单进程、单 GPU 训练
srun mpirun -n 1 lmp_mpi -in in.speed > log.lammps
