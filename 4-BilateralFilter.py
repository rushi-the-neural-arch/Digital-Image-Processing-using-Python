#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:10:32 2020

@author: rushirajsinh
"""

import cv2
from skimage import io
from skimage.filters import unsharp_mask


img_gaussian_noise = cv2.imread('globo-Scene-010.png', 0)
img_salt_pepper_noise = cv2.imread('globo-Scene-010.png', 0)

img = img_gaussian_noise

bilateral_using_cv2 = cv2.bilateralFilter(img, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('Bilateral.png', bilateral_using_cv2)

#d - diameter of each pixel neighborhood used during filtering
# sigmaCOlor - Sigma of grey/color space. 
#sigmaSpace - Large value means farther pixels influence each other (as long as the colors are close enough)

#
#from skimage.restoration import denoise_bilateral
#bilateral_using_skimage = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,
#                multichannel=False)

#sigma_color = float - Sigma for grey or color value. 
#For large sigma_color values the filter becomes closer to gaussian blur.
#sigma_spatial: float. Standard ev. for range distance. Increasing this smooths larger features.



#cv2.imshow("Original", img)
#cv2.imshow("cv2 bilateral", bilateral_using_cv2)
#cv2.imshow("Using skimage bilateral", bilateral_using_skimage)
#
#cv2.waitKey(0)          
#cv2.destroyAllWindows() 
unsharped_img = unsharp_mask(img, radius=3, amount=1)
io.imsave('Radius3amount1.png',unsharped_img)
#cv2.imwrite('Radius7amount1', unsharped_img)


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(unsharped_img, cmap='gray')
ax1.title.set_text('Gaussian Filter')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(bilateral_using_cv2, cmap='gray')
ax2.title.set_text('Bilateral Filter')

filename = 'Gaussian VS Bilateral.png'
  
# Using cv2.imwrite() method 
# Saving the image 
plt.savefig('Gaussian VS Bilateral.png', dpi=300, bbox_inches='tight')
plt.show()