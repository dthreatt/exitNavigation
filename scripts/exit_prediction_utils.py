"""
Utilities used during exit prediction
"""

import cv2
import numpy as np

def rotate_img(img,angle):
    img[img == -1] = 127 #so that the information is not lost when converting to unsigned int
    #print(np.unique(img),'before uint8')
    img=img.astype(np.uint8)
    #img[img == 100] = 255
    #print(np.unique(img),'after uint8')
    center=(img.shape[0]/2.,img.shape[1]/2.)
    #cv2.imshow('before rotation',img)
    #we used the flag inter nearest so that the map is still only 0, 100 and 127 after the rotation
    img = cv2.warpAffine(img,cv2.getRotationMatrix2D(center,angle,1),img.shape[0:2],flags=cv2.INTER_NEAREST,borderValue=127)
    #print(np.unique(img),'shaped map after rotation')
    img = img.astype(np.int8)
    #img[(img != 0) & (img != 101)] = 100 #this is because the warp affine interpolates pixel values, can use interpolation flag instead in the warp function
    #print(np.unique(img),'shaped map after rotation and tuning values')
    img[img == 127] = -1 #reintroduce the unknown values as -1
    #print(np.unique(img),'shaped map after rotation which should have -1, 0 and 100')
    #cv2.imshow('after rotation',img)
    #cv2.waitKey()
    return img