# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:00:31 2018

@author: ZSQ
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

sess = tf.InteractiveSession()

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

def oneHotDecoding(arrayCoded):
    """
    function:decoding for oneHot  
    input:the array coded of oneHot      
    output:the array decoded
    """
    arraySize = arrayCoded.shape
    arrayDecoded = np.zeros(arraySize[:2],dtype = 'int64')
    if arrayCoded.ndim == 3:   #(width, height, class)
        
        narray = np.max(arrayCoded, axis = 2)    
        
        for i in range(arraySize[2] - 1):
            narray = np.dstack((narray, np.max(arrayCoded, axis = 2)))
        if len(np.where(arrayCoded == narray)[2]) == arraySize[0] * arraySize[1]:
            arrayDecoded = np.where(arrayCoded == narray)[2].reshape(arraySize[:2])
        else:
            print('error:the max indexes of slice from arrayCoded have too many ')
    else:
        print("error:cannot broadcast shape {}".format(arrayCoded.shape))
        
    return arrayDecoded

def oneHotDecoding_(arrayCoded, thresh):
    """
    function:decoding for oneHot  
    input:the array coded of oneHot      
    output:the array decoded
    """
    arraySize = arrayCoded.shape
    arrayDecoded = np.zeros(arraySize[:2],dtype = 'int64')
    
    class_num = arraySize[2]
    if arrayCoded.ndim == 3:   #(width, height, class)
        if class_num == 1:
            arrayDecoded = (arrayCoded[:,:,0] >= thresh).astype(int)
        elif class_num == 2:
            arrayCoded[:,:,0] = (arrayCoded[:,:,0] >= thresh).astype(int)
            arrayCoded[:,:,1] = (arrayCoded[:,:,1] >= thresh).astype(int)*2
            arrayDecoded = arrayCoded[:,:,0]+arrayCoded[:,:,1]
        elif class_num == 3:
            arrayCoded[:,:,0] = (arrayCoded[:,:,0] >= thresh).astype(int)
            arrayCoded[:,:,1] = (arrayCoded[:,:,1] >= thresh).astype(int)*2
            arrayCoded[:,:,2] = (arrayCoded[:,:,2] >= thresh).astype(int)*3 
            arrayDecoded = arrayCoded[:,:,0]+arrayCoded[:,:,1]+arrayCoded[:,:,2]
        elif class_num == 4:
            arrayCoded[:,:,0] = (arrayCoded[:,:,0] >= thresh).astype(int)
            arrayCoded[:,:,1] = (arrayCoded[:,:,1] >= thresh).astype(int)*2
            arrayCoded[:,:,2] = (arrayCoded[:,:,2] >= thresh).astype(int)*3
            arrayCoded[:,:,3] = (arrayCoded[:,:,3] >= thresh).astype(int)*4
            arrayDecoded = arrayCoded[:,:,0]+arrayCoded[:,:,1]+arrayCoded[:,:,2]+arrayCoded[:,:,3]
        else:
            print('class num is wrong')
    else:
        print("error:cannot broadcast shape {}".format(arrayCoded.shape))
    return arrayDecoded
    

# 初始化数据
save_dir = ".\\train_dev_all\\"
path_testdata1 = save_dir + 'test_data_4.npz'
data_1 = np.load(path_testdata1)

X_test = data_1['test_data']
y_test = data_1['test_label']

###======================== HYPER-PARAMETERS ============================###
batch_size = 10
lr = 0.0001 
# lr_decay = 0.5
# decay_every = 100
beta1 = 0.9
n_epoch = 100
print_freq_step = 100
num_Classes = 1
task='twoClass_test'
thresh = 0.3


nw, nh, nz = X_test[0].shape

t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
## labels are either 0 or 1
t_seg = tf.placeholder('float32', [batch_size, nw, nh, num_Classes], name='target_segment')

## test inference
#net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
net_test = model.u_net_bn(t_image, is_train=False, reuse=False, n_out=num_Classes)

## test losses
test_out_seg = net_test.outputs
test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])


#with tf.Session() as sess:
tl.layers.initialize_global_variables(sess)

tl.files.load_and_assign_npz(sess, name=save_dir + 'u_net_twoClass.npz', network=net_test)

total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                batch_size=batch_size, shuffle=True):
    b_images, b_labels = batch
    y_testall = np.zeros(tuple(list(b_labels.shape[:]) + [num_Classes]), dtype=np.int32)

    if task == 'threeClass_test':
        y_testall[:,:,:,0] = (b_labels == 1).astype(int)
        y_testall[:,:,:,1] = (b_labels == 2).astype(int)
        y_testall[:,:,:,2] = (b_labels == 3).astype(int)
        y_testall[:,:,:,3] = (b_labels == 4).astype(int)  
            
    if task == 'twoClass_test':
        y_testall = (b_labels > 0).astype(int)
        y_testall = y_testall[:,:,:,np.newaxis]            
    
    _dice, _iou, _diceh, out = sess.run([test_dice_loss,
            test_iou_loss, test_dice_hard, net_test.outputs],
            {t_image: b_images, t_seg: y_testall})
    total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
    
    print('test_{} dice:{} iou:{} dice_hard:{}'.format(n_batch, _dice, _iou, _diceh))
    
    out_labels = np.zeros((out.shape[:3]))
    for k in range(batch_size):
        out_labels[k] = oneHotDecoding_(out[k], thresh)
    
#    plt.figure
#    plt.imshow(out[3][:,:,0])
#    plt.show()
#    
    ## save a predition of test set
    for i in range(batch_size):
        if np.max(b_images[i]) > 0:
            vis_imgs2(b_images[i], b_labels[i], out_labels[i], "samples/{}/test_{}.png".format(task, batch_size*n_batch+i))
            #break
        elif i == batch_size-1:
            vis_imgs2(b_images[i], b_labels[i], out_labels[i], "samples/{}/test_{}.png".format(task, batch_size*n_batch+i))
    n_batch += 1

print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
        (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))







