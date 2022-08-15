# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:57:47 2022

@author: cynth
"""

import cv2 as cv
import numpy as np

#Define function
def process(filename):
    img = cv.imread(filename)#Read image
    cropped = img[430:630, 1030:1370]#Crop image to the interested section
    b, g, r = cv.split(cropped)#Split colour channels
    blank = np.zeros(cropped.shape[:2], dtype='uint8')#Create a blank channel
    red = cv.merge([blank, blank, r])#Reconstruct image with only red pixels
    alpha = np.sum(red, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    processedimg = np.dstack((red, alpha))#Remove background
    cv.imshow('result', processedimg)
    cv.waitKey(0)
    
    return processedimg

process()