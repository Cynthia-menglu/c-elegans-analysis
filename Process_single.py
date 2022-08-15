# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:32:02 2022

@author: cynth
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image
img =  cv.imread('00009.jpg')
# cv.imshow('Original', img)

# 2. Crop image
cropped = img[430:630, 1030:1370]
# cv.imshow('Cropped', cropped)

# 3. Split colour channels
b, g, r = cv.split(cropped)
# cv.imshow('redsc', r)
# cv.imwrite('red.png', r)

# 4. Reconstruct BGR image with red pixels
blank = np.zeros(cropped.shape[:2], dtype='uint8')
red = cv.merge([blank, blank, r])
# cv.imshow('reconstructed red', red)

# 5. Remove background
# Load image as Numpy array in BGR order
# na = cv.imread('red.png')
# Make a True/False mask of pixels whose BGR values sum to more than zero
alpha = np.sum(red, axis=-1) > 0
# Convert True/False to 0/255 and change type to "uint8" to match "na"
alpha = np.uint8(alpha * 255)
# Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
res = np.dstack((red, alpha))
# cv.imshow('clean red', res)

# 6. Line fit
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299,0.587,0.114])

def feature(x, order = 1):
    #Generate linear feature of the form [1, x] where x is the column of x-coordinates and 1 is the column of ones for the intercept
    x = x.reshape(-1,1)
    return np.power(x, np.arange(order+1).reshape(1,-1))

I_orig = res.copy()
#Convert to grayscale
I = rgb2gray(I_orig)
# print(I)

#Add a mask if needed
mask = I>0
X = np.argwhere(mask)

#Get coordinates of pixels corresponding to the region
X = np.argwhere(I)
# print(X)

#Use the value as weights later
weights = I[mask]/float(I.max())
#Convert to diagonal matrix
W = np.diag(weights)
# print(W)

#Column indices
x=X[:,1].reshape(-1,1)
#Row indices to predict. Origins is at top left corner
y=X[:,0]

#Find vector w that minimise (Aw-y)^2 to predict y = wx
#least squares with l2 regularisation
#alpha = regularisation parameter; larger alpha => less flexible curve
alpha = 0.01

#Constuct data matrix, A
order = 1
A = feature(x, order)
#w = inv(A^T A + alpha*I) A^Ty
w, v = np.linalg.pinv(A.T.dot(A)+alpha*np.eye(A.shape[1])).dot(A.T).dot(y)
print('Derivative: ', v)

#Generate test points
n_samples = 50
x_test = np.linspace(0, I_orig.shape[1], n_samples)
X_test = feature(x_test, order)
#predict y coordinates
y_test = X_test.dot([w, v])
#Display
fig, ax = plt.subplots(1, 1, figsize = (10,5))
ax.imshow(I_orig)
ax.plot(x_test, y_test, color = 'green')
# fig.legend()
fig.savefig('linefit.png')



cv.waitKey(0)