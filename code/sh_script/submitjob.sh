#!/usr/bin/env bash

exp='run_exp_all.py'

ln -s ../$exp $exp
chmod +x $exp

script='run_exp.sh'

# # mysseg
# for rand in 0 1 3 5 6
# do
# 	for model in 'multi_srm' 'all_srm' 'indv_srm' 'all_ica' 'indv_ica' 'all_gica' 'indv_gica' 'all_dict' 'indv_dict' 'avg'
# 	do
# 		sbatch $script 'mysseg' $model $rand
# 	done
# done

# # mysseg_all
# for rand in 0 1 3 5 6
# do
# 	for model in 'multi_srm'
# 	do
# 		sbatch $script 'mysseg_all' $model $rand
# 	done
# done

# # mapping
# for rand in 0 1 3 5 6
# do
# 	for model in 'multi_srm' 'all_srm' 'all_ica' 'all_gica' 'all_dict' 'avg'
# 	do
# 		sbatch $script 'mapping' $model $rand
# 	done
# done

# # mapping_loo
# for rand in 0 1 2 3 4
# do
# 	for model in 'multi_srm' 'all_srm' 'all_ica' 'all_gica' 'all_dict'
# 	do
# 		sbatch $script 'mapping_loo' $model $rand
# 	done
# done
# sbatch $script 'mapping_loo' 'avg' 0

# # loods
# for rand in 0 1 2 3 4
# do
# 	for model in 'multi_srm'
# 	do
# 		sbatch $script 'loods' $model $rand
# 	done
# done


# # overfit
# for rand in 0 1 2 3 4
# do
# 	for model in 'indv_srm' 'indv_ica' 'indv_gica' 'indv_dict'
# 	do
# 		sbatch $script 'overfit' $model $rand
# 	done
# done
