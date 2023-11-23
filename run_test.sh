#!/bin/bash
#SBATCH -o errors.txt
#SBATCH -J wav2vec2
#SBATCH -G a100:1

python -u test.py > logs/wav2vec2_test.txt
