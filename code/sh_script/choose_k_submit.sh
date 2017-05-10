#!/usr/bin/env bash

exp='choose_k_exp.py'

ln -s ../$exp $exp
chmod +x $exp

script='choose_k.sh'

# # mysseg
# for nfeat in  25 50 75 100
# do
# 	for model in 'multi_dict' 'indv_dict'
# 	do
# 		for roi in 'dmn' 'pt' 'eac'
# 		do
# 			sbatch $script 'mysseg' $model $nfeat $roi
# 		done
# 	done
# done

# # mapping
# for nfeat in 25 50 75 100
# do
# 	for model in 'multi_dict' 'indv_dict'
# 	do
# 		for roi in 'dmn' 'pt' 'eac'
# 		do
# 			sbatch $script 'mapping' $model $nfeat $roi
# 		done
# 	done
# done

# imgpred
for nfeat in 25 50 75 100
do
	for model in 'multi_dict' 'indv_dict'
	do
		sbatch $script 'imgpred' $model $nfeat 'pmc'
	done
done
