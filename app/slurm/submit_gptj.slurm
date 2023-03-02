#!/bin/bash
#SBATCH --job-name=cs324-finetune_gptj
#
#SBATCH --partition=sphinx
#SBATCH --exclude=sphinx[1-3]
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/exe_log/finetune_gptj_%j.log
#SBATCH --account=nlp



source activate base
cd /afs/cs.stanford.edu/u/biyuan/fm/cs324/finetune_src
nvidia-smi
set -o allexport && source .env
python app/serve_finetune_gptj.py