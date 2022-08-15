# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:52:48 2022

@author: cynth
"""

import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

def process(image):
    img = cv.imread(image) # Read image
    cropped = img[350:700, 900:1240] # Crop image
    b, g, r = cv.split(cropped) # Split colour channels
    blank = np.zeros(cropped.shape[:2], dtype='uint8') # Create blank channels
    red = cv.merge([blank, blank, r]) # Reconstruct red pixels
    alpha = np.sum(red, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    processed = np.dstack((red, alpha)) # Remove background
    cv.waitKey(0)
    return processed

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299,0.587,0.114])

def feature(x, order = 1):
    #Generate linear feature of the form [1, x] where x is the column of x-coordinates and 1 is the column of ones for the intercept
    x = x.reshape(-1,1)
    return np.power(x, np.arange(order+1).reshape(1,-1))

list = []
for file_name in glob.iglob("*.jpg", recursive=True):
    I_orig = process(file_name)
    I = rgb2gray(I_orig)
    X = np.argwhere(I)
    x=X[:,1].reshape(-1,1)
    y=X[:,0]
    alpha = 0.01
    order = 1
    A = feature(x, order)
    w, v = np.linalg.pinv(A.T.dot(A)+alpha*np.eye(A.shape[1])).dot(A.T).dot(y)
    angle = np.rad2deg(np.arctan2([v], [1]))
    if angle < 0:
        true_angle = -(90+angle)
    elif angle > 0:
        true_angle = 90-angle 
    elif angle == 0:
        true_angle = 0
    else:
        true_angle = '_'
    result = list.append(true_angle)

# print(type(list))        
print(list)




