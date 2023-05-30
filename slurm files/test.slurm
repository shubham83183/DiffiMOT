#!/bin/bash
#SBATCH --job-name=main
#SBATCH -o /work/ws-tmp/g051507-thesis1/Diff_matching/Results_withoutInf_1000_5000ABS02/CrowdhumanLambda03.txt
#SBATCH --time=23:59:59
#SBATCH --mem=64G
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
cd  /work/ws-tmp/g051507-thesis1/Diff_matching/trackformer
conda activate  Diff_matching
python -m torch.distributed.launch --nproc_per_node=1 --use_env src/train.py with \
            crowdhuman \
            deformable \
            multi_frame \
            tracking \
            output_dir=models/crowdhuman_deformable_multi_frame_WithoutInf_lambda1000_5000Abs02 \

