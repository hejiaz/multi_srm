#!/usr/bin/env bash

exp='run_exp_all.py'

ln -s ../$exp $exp
chmod +x $exp

script='run_exp.sh'

# # mysseg
# for rand in 0 2 6 7 8
# do
# 	for model in 'all_srm'
# 	do
# 		sbatch $script 'mysseg' $model $rand
# 	done
# done

# # mysseg_all
# for rand in 0 2 6 7 8
# do
# 	for model in 'multi_srm' 'multi_dict'
# 	do
# 		sbatch $script 'mysseg_all' $model $rand
# 	done
# done

# # mapping
# for rand in 0 2 6 7 8
# do
# 	for model in 'all_srm'
# 	do
# 		sbatch $script 'mapping' $model $rand
# 	done
# done

# # imgpred
# for rand in 0 1 2 3 4 
# do
# 	for model in 'all_srm'
# 	do
# 		sbatch $script 'imgpred' $model $rand
# 	done
# done

# # mapping_all
# for rand in 0 1 3 5 6
# do
# 	for model in 'multi_srm' 'all_srm' 'all_ica' 'all_gica' 'all_dict' 'avg'
# 	do
# 		sbatch $script 'mapping_all' $model $rand
# 	done
# done

# # mapping_loo
# for rand in 0 1 2 3 4
# do
# 	# for model in 'multi_srm' 'all_srm' 'all_ica' 'all_gica' 'all_dict'
# 	for model in 'all_dict'
# 	do
# 		sbatch $script 'mapping_loo' $model $rand
# 	done
# done
# sbatch $script 'mapping_loo' 'avg' 0

# # loods
# for rand in 0 1 2 3 4
# do
# 	for model in 'multi_dict'
# 	do
# 		sbatch $script 'loods' $model $rand
# 	done
# done


# # overfit
# for rand in 0 1 2 3 4
# do
# 	# for model in 'indv_srm' 'indv_ica' 'indv_gica' 'indv_dict'
# 	for model in 'indv_dict'
# 	do
# 		sbatch $script 'overfit' $model $rand
# 	done
# done


# # shared_subj
# for rand in 0 1 2 3 4 5 6 7
# do
# 	for model in 'multi_srm'
# 	do
# 		sbatch $script 'shared_subj' $model $rand
# 	done
# done

# # dist_subj
# for rand in 0 1 2 3 4
# do
# 	for model in 'multi_srm'
# 	do
# 		for ds in '1,0' '2,0'
# 		do
# 			for shared in 10 12
# 			do
# 				sbatch $script 'dist_subj' $model $rand $shared $ds
# 			done
# 		done
# 	done
# done


# different_TR
for rand in 0 2 6 7 8
do 
	for model in 'multi_srm'
	do
		for portion in 0.2 0.4 0.6 0.8
		do
			sbatch $script 'different_TR' $model $rand $portion
		done
	done
done

