#!/bin/bash
#SBATCH --job-name=spiking_resformer
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=<account_name>

module load python/3.8
module load cuda/11.4
module load pytorch/1.9.0-cuda11.1

source /path/to/your/venv/bin/activate

SCRIPT_PATH="train_model.py"
CONFIG_PATH="config.yaml"

export CUDA_VISIBLE_DEVICES=0

python $SCRIPT_PATH --config $CONFIG_PATH

