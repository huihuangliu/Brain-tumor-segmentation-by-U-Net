# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:22:11 2018

@author: ZSQ
"""

# unaccomplished
def oneHotDecoding(arrayCoded):
    """
    function:decoding for oneHot  
    input:the array coded of oneHot      
    output:the array decoded
    """
    arraySize = arrayCoded.shape
    arrayDecoded = np.zeros(arraySize[:2],dtype = 'int32')
    if arrayCoded.ndim == 3:   #(width, height, class)
        
        narray = np.max(arrayCoded, axis = 2)
        scalarCoded = np.zeros((arraySize[2], arraySize[0] * arraySize[1]))
        
#        for k in range(arraySize[2]):
#            scalarCoded[k, :] = arrayCoded[:,:,k].reshape(1,-1)
        scalarCoded = arrayCoded.reshape(((arraySize[2], arraySize[0] * arraySize[1])))
            
        if np.where(scalarCoded == np.max(scalarCoded, axis = 1)):
            
        
        for i in range(arraySize[2] - 1):
            narray = np.dstack((narray, np.max(arrayCoded, axis = 2)))
        if np.where(arrayCoded == narray)[2] == arraySize[0] * arraySize[0]
            arrayDecoded = np.where(arrayCoded == narray)[2].reshape(arraySize[:2])
        else:
            
            scalarCoded
    else:
        print("cannot broadcast shape {}".format(arrayCoded.shape))
        
    return arrayDecoded

array = np.zeros((3, 3, 4),dtype = 'int32')
array[:,:,0] = [[0,1,0],[1,0,0],[0,0,1]]
array[:,:,1] = [[1,0,0],[0,0,1],[0,0,0]]
array[:,:,2] = [[0,0,1],[0,0,0],[1,0,0]]
array[:,:,3] = [[0,0,0],[0,1,0],[0,1,0]]

arrayDecoded = oneHotDecoding(array)

import numpy as np
arrayCoded = tmp
arraySize = arrayCoded.shape
arrayDecoded = np.zeros(arraySize[:2],dtype = 'int64')
if arrayCoded.ndim == 3:   #(width, height, class)
    
#    narray = np.ones((arraySize[:2]),dtype = 'int64') * 0.5    
#    
#    for i in range(arraySize[2] - 1):
#        narray = np.dstack((narray, np.max(arrayCoded, axis = 2)))
#    if len(np.where(arrayCoded >= narray)[2]) == arraySize[0] * arraySize[1]:
#        arrayDecoded = np.where(arrayCoded >= narray)[2].reshape(arraySize[:2])
#    else:
#        print('error:the max indexes of slice from arrayCoded have too many ')
    narray = np.ones((arraySize[:2]),dtype = 'int64') * 0.5
    for i in range(arraySize[2]):
        arrayCoded[np.where(arrayCoded[:,:,i] >= narray)] = i
    a = arrayCoded[:,:,0] < narray
    b = arrayCoded[:,:,1] < narray
    c = arrayCoded[:,:,2] < narray
    d = arrayCoded[:,:,3] < narray
    coordinates = []
    for i in range(arraySize[0]):
        for j in range(arraySize[1]):
            if (a[i,j] and b[i,j] and c[i,j] and d[i,j]) == True:
                coordinates.append((i,j))
    arrayCoded[coordinates] = 0
else:
    print("error:cannot broadcast shape {}".format(arrayCoded.shape))
    






