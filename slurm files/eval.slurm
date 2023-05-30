#!/bin/bash
#SBATCH --job-name=eval2
#SBATCH -o /work/ws-tmp/g051507-thesis/g051507-thesis-1679703002/Diff_matching/Results_withoutinf_0.001/mot17_private_eval124_all.txt
#SBATCH --time=04:00:59
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

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
cd  /work/ws-tmp/g051507-thesis/g051507-thesis-1679703002/Diff_matching/trackformer
conda activate  Diff_matching
python src/track.py with reid



