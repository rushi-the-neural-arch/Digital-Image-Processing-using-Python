#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:51:26 2020

@author: rushirajsinh
"""

import matplotlib.pyplot as plt
from skimage import io

import numpy as np
from skimage.filters import threshold_otsu
import cv2

img = io.imread("Bigmouth_frame157.png", as_gray=True)

############  ENTROPY  ##################


from skimage.filters.rank import entropy
from skimage.morphology import disk
entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img)


#use otsu to threshold high vs low entropy regions.

# min = 0, max = 5 for this PARTICULAR image, (entropy_img.min())

plt.hist(entropy_img.flat, bins=100, range=(0,5))  #.flat returns the flattened numpy array (1D)

thresh = threshold_otsu(entropy_img) 

#binarize the entropy image 
binary = entropy_img <= thresh
plt.imshow(binary)