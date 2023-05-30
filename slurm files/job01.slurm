#!/bin/bash
#SBATCH --job-name=Diffitrack_private01
#SBATCH -o /work/ws-tmp/g051507-Thesis_april/g051507-thesis-1682550603/g051507-thesis-1679703002/Diff_matching/Results_withoutInf_1000_5000_abs_mean/mot17_private01.txt
#SBATCH --time=23:59:59
#SBATCH --mem=220G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

module load GpuModules
module load cuda10.2/toolkit/10.2.89
module unload tensorflow2-py37-cuda11.2-gcc8/2.5.2
module unload protobuf3-gcc8/3.8.0nccl2-cuda11.2-gcc8/2.11.4
module unload cudnn8.1-cuda11.2/8.1.1.33  
module unload protobuf3-gcc8/3.8.0
module unload cuda11.2/toolkit/11.2.2
module unload openblas/dynamic/0.3.7
module unload hdf5_18/1.8.21

source ~/.bashrc
conda deactivate
cd  /work/ws-tmp/g051507-Thesis_april/g051507-thesis-1682550603/g051507-thesis-1679703002/Diff_matching/trackformer
conda activate  Diff_matching

ip1=`hostname -I | awk '{print $2}'`
echo $ip1

export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

python -m  torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=$ip1 --master_port=8999 --use_env\
            src/train.py with \
            crowdhuman \
            deformable \
            multi_frame \
            tracking \
            output_dir=models/mot17_crowdhuman_deformable_multi_frame_WithoutInf_lambda1000_5000Abs_mean \



