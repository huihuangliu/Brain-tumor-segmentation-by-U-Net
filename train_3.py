#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model

 

def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y

def data_augmentation(data):
    """ data augumentation """
    x1, x2, x3, x4, y1, y2, y3, y4 = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y1, y2, y3, y4],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y1, y2, y3, y4],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.rotation_multi([x1, x2, x3, x4, y1, y2, y3, y4], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.shift_multi([x1, x2, x3, x4, y1, y2, y3, y4], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.shear_multi([x1, x2, x3, x4, y1, y2, y3, y4], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y1, y2, y3, y4 = tl.prepro.zoom_multi([x1, x2, x3, x4, y1, y2, y3, y4],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y1, y2, y3, y4
    

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


def main(task='threeClass'):
    ## Create folder to save trained model and result images
    '''
    save_dir = "./checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("./samples/{}".format(task))
    '''
    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    #import prepare_data_with_valid_3 as dataset
    #X_train = dataset.X_train_input
    #y_train = dataset.X_train_target[:,:,:,np.newaxis]
    #X_test = dataset.X_dev_input
    #y_test = dataset.X_dev_target[:,:,:,np.newaxis]
    
    '''
    print('loading data....')
    save_dir = ".\\train_dev_all\\"
    path_traindata1 = save_dir + 'train_data_1.npz'
    data1 = np.load(path_traindata1)
    
    path_traindata2 = save_dir + 'train_data_2.npz'
    data2 = np.load(path_traindata2)
    
    X_train = np.concatenate((data1['train_data'],data2['train_data']),axis=0)
    y_train = np.concatenate((data1['train_label'],data2['train_label']),axis=0)
            
    path_testdata1 = save_dir + 'test_data_1.npz'
    data_1 = np.load(path_testdata1)
    path_testdata2 = save_dir + 'test_data_2.npz'
    data_2 = np.load(path_testdata2)
    X_test = np.concatenate((data_1['test_data'],data_2['test_data']),axis=0)
    y_test = np.concatenate((data_2['test_label'],data_2['test_label']),axis=0)

    print('loading data done')
    
    '''
    print('loading data....')
    save_dir = ".\\train_dev_all\\"
    path_traindata1 = save_dir + 'train_data_4.npz'
    data1 = np.load(path_traindata1)
      
    X_train = data1['train_data']
    y_train = data1['train_label']
            
    path_testdata1 = save_dir + 'test_data_3.npz'
    data_1 = np.load(path_testdata1)
    X_test = data_1['test_data']
    y_test = data_1['test_label']

    print('loading data done')
    
    
    '''
    y_trainall = np.zeros(tuple(list(y_train.shape[:]) + [4]), dtype=np.int32)
    y_testall = np.zeros(tuple(list(y_test.shape[:]) + [4]), dtype=np.int32)
    
    
    if task == 'threeClass':
        y_trainall[:,:,:,0] = (y_train == 1).astype(int)
        y_trainall[:,:,:,1] = (y_train == 2).astype(int)
        y_trainall[:,:,:,2] = (y_train == 3).astype(int)
        y_trainall[:,:,:,3] = (y_train == 4).astype(int)
        
        y_testall[:,:,:,0] = (y_test == 1).astype(int)
        y_testall[:,:,:,1] = (y_test == 2).astype(int)
        y_testall[:,:,:,2] = (y_test == 3).astype(int)
        y_testall[:,:,:,3] = (y_test == 4).astype(int)
    '''
    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001 
    lr_decay = 0.5
    decay_every = 100
    beta1 = 0.9
    n_epoch = 100
    print_freq_step = 100
    num_Classes = 1
    thresh = 0.3
    
    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    '''
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis],
                X[:,:,2,np.newaxis], X[:,:,3,np.newaxis], y[:,:,np.newaxis]])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    '''
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    ###======================== DEFIINE MODEL =======================###
    ## nz is 4 as we input all Flair, T1, T1c and T2.
    t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
    ## labels are either 0 or 1
    t_seg = tf.placeholder('float32', [batch_size, nw, nh, num_Classes], name='target_segment')
    ## train inference
    #net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
    #net = model.u_net(t_image, is_train=True, reuse=False, n_out=num_Classes)
    net = model.u_net_bn(t_image, is_train=True, reuse=False, n_out=num_Classes)
    ## test inference
    #net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
    #net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=num_Classes)
    net_test = model.u_net_bn(t_image, is_train=False, reuse=True, n_out=num_Classes)

    ###======================== DEFINE LOSS =========================###
    ## train losses
    out_seg = net.outputs
    dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
    dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
    loss = dice_loss

    ## test losses
    test_out_seg = net_test.outputs
    test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
    test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

    ###======================== DEFINE TRAIN OPTS =======================###
    t_vars = tl.layers.get_variables_with_name('u_net', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
        train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

    ###======================== LOAD MODEL ==============================###
    tl.layers.initialize_global_variables(sess)
    ## load existing model if possible
    #tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

    ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):   #100
        epoch_time = time.time()
        ## update decay learning rate at the beginning of a epoch
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr))
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            '''
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels[:,:,:,np.newaxis])],
                    fn=distort_imgs) # (10, 5, 240, 240, 1)
            '''
            '''
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels[:,:,:,0, np.newaxis], labels[:,:,:,1, np.newaxis],
                    labels[:,:,:,2, np.newaxis], labels[:,:,:,3, np.newaxis])],fn=data_augmentation) # (10, 5, 240, 240, 1)
            b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,0:4,:,:,:]
    
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, nw, nh, nz)
            
            b_labels = b_labels.transpose((0,2,3,1,4))
            b_labels.shape = (batch_size, nw, nh, nz)
            
            new_iamges = np.concatenate((images, b_images), axis=0)
            new_labels = np.concatenate((labels, b_labels), axis=0)
            '''
            
            b_images = images
            b_labels = np.zeros(tuple(list(labels.shape[:]) + [num_Classes]), dtype=np.int32)
            if task == 'threeClass':
                
                b_labels[:,:,:,0] = (labels == 1).astype(np.int32)
                b_labels[:,:,:,1] = (labels == 2).astype(np.int32)
                b_labels[:,:,:,2] = (labels == 3).astype(np.int32)
                b_labels[:,:,:,3] = (labels == 4).astype(np.int32)
                #b_labels[:,:,:,4] = (labels == 4).astype(np.int32)
            if task == 'twoClass':
                b_labels = (labels > 0).astype(np.int32)
                b_labels = b_labels[:,:,:,np.newaxis]
                
                
            ## update network
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            ## you can show the predition here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            # if _dice == 1: # DEBUG
            #     print("DEBUG")
            #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))
        
        out_labels = np.zeros((out.shape[:3]))
        for k in range(batch_size):
            out_labels[k] = oneHotDecoding_(out[k], thresh)
        
        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], labels[i], out_labels[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], labels[i], out_labels[i], "samples/{}/train_{}.png".format(task, epoch))
        
        
        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            y_testall = np.zeros(tuple(list(labels.shape[:]) + [num_Classes]), dtype=np.int32)
            if task == 'threeClass':
                y_testall[:,:,:,0] = (b_labels == 1).astype(int)
                y_testall[:,:,:,1] = (b_labels == 2).astype(int)
                y_testall[:,:,:,2] = (b_labels == 3).astype(int)
                y_testall[:,:,:,3] = (b_labels == 4).astype(int)  
            
            if task == 'twoClass':
                y_testall = (b_labels > 0).astype(int)
                y_testall = y_testall[:,:,:,np.newaxis]
            
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: y_testall})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        print(" task: {}".format(task))
        
        out_labels = np.zeros((out.shape[:3]))
        for k in range(batch_size):
            out_labels[k] = oneHotDecoding_(out[k], thresh)
        
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out_labels[i], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out_labels[i], "samples/{}/test_{}.png".format(task, epoch))
        
        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 3'
    '''
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')

    args = parser.parse_args()

    main(args.task)
    '''
    print('starting ')
    task = 'twoClass'
    main(task)
