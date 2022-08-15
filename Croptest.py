# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:59:59 2022

@author: cynth
"""

import cv2 as cv

img = cv.imread('01785.jpg')
img_cropped = img[350:700, 900:1240]

cv.imshow('cropped', img_cropped)

cv.waitKey(0)