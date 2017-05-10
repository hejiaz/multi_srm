#!/usr/bin/env bash
# J--job name
# o--output file
# t--time limit (minutes)

#SBATCH -N 1
#SBATCH -J 'multi_srm'
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%j.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=hejiaz@princeton.edu
#SBATCH -t 240
#SBATCH --mem=48000

./choose_k_exp.py $1 $2 $3 $4

###--mem=65536