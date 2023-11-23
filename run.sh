#!/bin/bash
#SBATCH -o errors.txt
#SBATCH -J wav2vec2
#SBATCH -G 1
#SBATCH -x xgph[0-19],xgpd[0-9],xgpc[0-9],xgpf[0-11],xgpb[0-2]
#SBATCH --partition=long
#SBATCH --time=4320

python -u train.py > logs/hubert_baseline_logs.txt
