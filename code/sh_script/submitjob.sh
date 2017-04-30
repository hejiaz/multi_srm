#!/usr/bin/env bash

exp='run_exp_all.py'

ln -s ../$exp $exp
chmod +x $exp

script='run_exp.sh'
expt='mysseg'

for nfeat in 25 50 75 100 125 150 200
do
	for model in 'indv_srm' 'indv_ica' 'indv_gica' 'indv_dict' 'multi_srm'
	do
		sbatch $script $expt $model $nfeat
	done
done

