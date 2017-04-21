#!/usr/bin/env python
# By Hejia Zhang @ Princeton

import nibabel as nib
import numpy as np 

wb_file = '/tigress/hejiaz/data/mask/MNI152_T1_3mm_brain_mask.nii'
mask_file = '/tigress/hejiaz/data/mask/PT_thr30.nii.gz'
mask_out = '/tigress/hejiaz/data/mask/pt_3d_mask.nii.gz'

mask = nib.load(mask_file)
mask_data = mask.get_data()
wb = nib.load(wb_file)
affine = wb.get_affine()
wb_data = wb.get_data()

new_mask = np.zeros_like(wb_data)
for i in range(61):
	for j in range(73):
		for k in range(61):
			if mask_data[i,j,k] > 2.4:
				new_mask[i,j,k] = 1

print (sum(sum(sum(new_mask))))

mask_img = nib.Nifti1Image(new_mask,affine)
nib.save(mask_img,mask_out)