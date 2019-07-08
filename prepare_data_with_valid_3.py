#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:37:36 2018

@author: liuhuihuang
"""

import glob
import os
import fnmatch
from PIL import Image
import SimpleITK as sitk
import numpy as np
#import pylab
import matplotlib.pyplot as plt
import cv2
import pickle
import tensorflow as tf
import gc


train_data_path = "./BRATS-Training/**/**/**/*.mha"
save_dir = "./train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

all_data_name = glob.glob(train_data_path)  #读取文件路径
file_names_t1 = []
file_names_t2 = []
file_names_t1c = []
file_names_flair = []
file_names_gt = []

# we suppose that images are read in sequence. Afte 4 files, 5th is ground truth
for i in range(len(all_data_name)):
    fname = os.path.basename(all_data_name[i])
    if fnmatch.fnmatch(fname, '*T1c.*'):
        file_names_t1c.append(all_data_name[i])
    elif fnmatch.fnmatch(fname, '*T1.*'):
        file_names_t1.append(all_data_name[i])
    elif fnmatch.fnmatch(fname, '*T2.*'):
        file_names_t2.append(all_data_name[i])
    elif fnmatch.fnmatch(fname, '*Flair.*'):
         file_names_flair.append(all_data_name[i])
    elif fnmatch.fnmatch(fname, '*_3more*'):
         file_names_gt.append(all_data_name[i])
         
#split train data and test data

flies_all_name = [file_names_flair, file_names_t1, file_names_t1c, file_names_t2]

#LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD    
data_types = ['flair', 't1', 't1c', 't2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

num_patients = len(flies_all_name[0])
for i, name in enumerate(data_types):
    data_temp_list = []
    for j in range(num_patients):
        img_path = flies_all_name[i][j]
        image = sitk.ReadImage(img_path)               
        img_arr = sitk.GetArrayFromImage(image)  # 176*240*240   z*y*x
        data_temp_list.append(img_arr)

    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[name]['mean'] = m
    data_types_mean_std_dict[name]['std'] = s
del data_temp_list
print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)
    
#get normalize image

# X_train_target_whole = [] # 1 2 4
# X_train_target_core = [] # 1 4
# X_train_target_enhance = [] # 4

X_dev_input = []
X_dev_target = []
# X_dev_target_whole = [] # 1 2 4
# X_dev_target_core = [] # 1 4
# X_dev_target_enhance = [] # 4

print(" train data generating")
half = [int(num_patients/2), num_patients]
#for time in range(2):
for time in [1]:
    X_train_input = []
    X_train_target = []
    for i in range(half[time]):
        all_3d_data = []
        for j, name in enumerate(data_types):
            img_path = flies_all_name[j][i]
            image = sitk.ReadImage(img_path)               
            img_arr = sitk.GetArrayFromImage(image)
            img_arr = (img_arr - data_types_mean_std_dict[name]['mean']) / data_types_mean_std_dict[name]['std']
            img = img_arr.astype(np.float32)
            all_3d_data.append(img)
    
        seg_path = file_names_gt[i]
        seg_img = sitk.ReadImage(seg_path)               
        seg_img = sitk.GetArrayFromImage(seg_img)
        #seg_img = np.transpose(seg_img, (1, 0, 2))
        #seg_img = np.transpose(seg_img, (2, 1, 0))
        for j in range(20,all_3d_data[0].shape[0]-20):
            combined_array = np.stack((all_3d_data[0][j, :, :], all_3d_data[1][j, :, :], all_3d_data[2][j, :, :], all_3d_data[3][j, :, :]), axis=0)
            #combined_array = np.transpose(combined_array, (1, 0, 2))
            combined_array = np.transpose(combined_array, (2, 1, 0))#.tolist()
            combined_array.astype(np.float32)
            X_train_input.append(combined_array) 
    
            seg_2d = seg_img[j, :, :]
            #seg_2d = np.transpose(seg_2d, (2, 1, 0))
            seg_2d.astype(int)
            X_train_target.append(seg_2d)
        del all_3d_data
        gc.collect()
        print("finished {}".format(i))
        
    X_train_input = np.asarray(X_train_input, dtype=np.float32)
    X_train_target = np.asarray(X_train_target)
    
    num = np.shape(X_train_target)[0]
    '''
    index = np.arange(num)
    np.random.shuffle(index)
    X_train_input = X_train_input[index, :, :, :]
    #lab = lab[index, :, :, :]
    X_train_target = X_train_target[index, :, :]
     '''  
     
    trainimg = X_train_input
    trainlab = X_train_target
    #split the data to trainimg and testimg
#    trainimg = X_train_input[:(num-20), :, :, :]
#    trainlab = X_train_target[:(num-20), :, :]
#    #trainlab = lab[:int(num * 0.9), :, :]
#    testimg = X_train_input[-20:, :, :, :]
#    testlab = X_train_target[-20:, :, :]
#    #testlab = lab[-int(num * 0.1):, :, :]
    del X_train_input, X_train_target
    
    print('start to save data')
    path_traindata = save_dir + 'train_data_' + str(4) + '.npz'
    np.savez(path_traindata, train_data = trainimg, train_label = trainlab)
#    path_testdata = save_dir + 'test_data_' + str(4) + '.npz'
#    np.savez(path_testdata, test_data = testimg, test_label = testlab)
    print('save data done')

'''
print('start to save data')

with open(save_dir + 'train_input.pickle', 'wb') as f:
    pickle.dump(trainimg, f, protocol=4)
with open(save_dir + 'train_target.pickle', 'wb') as f:
    pickle.dump(trainlab, f, protocol=4)

with open(save_dir + 'test_input.pickle', 'wb') as f:
    pickle.dump(testimg, f, protocol=4)
with open(save_dir + 'test_target.pickle', 'wb') as f:
    pickle.dump(testlab, f, protocol=4)

print('save data done')
'''