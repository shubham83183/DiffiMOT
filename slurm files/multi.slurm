#!/bin/bash
#SBATCH --job-name=list
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb

export MASTER_PORT=8990
export WORLD_SIZE=8

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/.bashrc
conda deactivate
cd  /work/ws-tmp/g051507-thesis1/trackformer
conda activate  TRACKFORMER1

### the command to run
python -m  torch.distributed.launch --nproc_per_node=4 --nnodes=2  --master_addr=$ip1 --master_port=8999 --use_env\
            src/train.py with \
            crowdhuman\
            deformable\
            multi_frame\
            tracking\
            output_dir=models/crowdhuman_deformable_multi_frame\