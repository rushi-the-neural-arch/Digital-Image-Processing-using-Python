#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:28:32 2020

@author: rushirajsinh
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu
import cv2

from skimage import io
from skimage.filters import unsharp_mask


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
thresholds = threshold_multiotsu(img, classes=5)

# Digitize (segment) original image into multiple classes.
#np.digitize assign values 0, 1, 2, 3, ... to pixels in each class.
regions = np.digitize(img, bins=thresholds)
plt.imshow(regions)
plt.imsave("BigMouth-5.png", regions)


segm1 = (regions == 0)
segm2 = (regions == 1)
segm3 = (regions == 2)
segm4 = (regions == 3)
segm5 = (regions == 4)

 

#We can use binary opening and closing operations to clean up. 
#Open takes care of isolated pixels within the window
#Closing takes care of isolated holes within the defined window

from scipy import ndimage as nd

segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))

segm5_opened = nd.binary_opening(segm5, np.ones((3,3)))
segm5_closed = nd.binary_closing(segm5_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((img.shape[0], img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)
all_segments_cleaned[segm5_closed] = (1,1,1)


plt.imshow(all_segments_cleaned)  #All the noise should be cleaned now
plt.imsave("BigMouth-5-Open_close.png", all_segments_cleaned) 