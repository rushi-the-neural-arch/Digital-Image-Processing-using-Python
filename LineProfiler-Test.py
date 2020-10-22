#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:03:09 2020

@author: rushirajsinh
"""
from line_profiler import LineProfiler
import random


from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu
import cv2

from skimage import io
from skimage.filters import unsharp_mask

# def do_stuff(numbers):
#     s = sum(numbers)
#     l = [numbers[i]/43 for i in range(len(numbers))]
#     m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

# numbers = [random.randint(1,100) for i in range(1000)]
# lp = LineProfiler()
# lp_wrapper = lp(do_stuff)
# lp_wrapper(numbers)
# lp.print_stats()

img = cv2.imread("Bigmouth_frame157.png", 0)

small = cv2.resize(img, ((300, 300)))

#Denoise for better results
from skimage.restoration import denoise_tv_chambolle
denoised_img = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
#denoised_img = unsharp_mask(img, radius=3, amount=2)

plt.imshow(denoised_img, cmap='gray')
plt.hist(denoised_img.flat, bins=100, range=(100,255))

plt.imshow(img, cmap='gray')
plt.hist(img.flat, bins=100, range=(100,255))  #.flat returns the flattened numpy array (1D)

####AUTO###########################
# Apply multi-Otsu threshold 
# thresholds = threshold_multiotsu(img, classes=5)

def multiotsu(classes):
    thresholds = threshold_multiotsu(img, classes=classes)
    
lp = LineProfiler()
lp_wrapper = lp(multiotsu)
lp_wrapper(5)
lp.print_stats()
