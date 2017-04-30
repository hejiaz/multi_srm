#!/usr/bin/env bash
# J--job name
# o--output file
# t--time limit (minutes)

#SBATCH -N 1
#SBATCH -J 'multi_srm'
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%j.out
#SBATCH -t 300

./run_exp_all.py $1 $2 $3

