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
#SBATCH -t 200
#SBATCH --mem=48000

./run_exp_all.py $1 $2 $3 -p $4  #-s $4 -d $5

###--mem=65536