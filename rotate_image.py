#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:43:13 2018

@author: wangxuchu
"""
import tensorlayer as tl
import numpy as np
import SimpleITK as sitk


img_path = './BRATS-Training/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512/VSD.Brain.XX.O.MR_Flair.54512.mha'
image_dir = './save_image'  
tl.files.exists_or_mkdir(image_dir)
image = sitk.ReadImage(img_path)               
img_arr = sitk.GetArrayFromImage(image)  # 176*240*240   z*y*x
img_arr = np.asarray(img_arr[90])
rotate_img = tl.prepro.flip_axis(img_arr, axis=1, is_random=False)
tl.vis.save_images(rotate_img[np.newaxis,:,:], size=(1, 1), image_path = 'save_image/train_im_rot.png')
tl.vis.save_images(img_arr[np.newaxis,:,:], size=(1, 1), image_path = 'save_image/train_im_ori.png')

synthesis = img_arr + rotate_img

tl.vis.save_images(synthesis[np.newaxis,:,:], size=(1, 1), image_path = 'save_image/train_im_syn.png')
