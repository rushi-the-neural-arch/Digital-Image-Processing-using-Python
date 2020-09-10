#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:11:39 2020

@author: rushirajsinhparmar
"""

"""

from skimage import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian

img = img_as_float(io.imread("/Users/rushirajsinhparmar/Downloads/DIP-Vectorly/globo-Scene-010.png", as_gray=True))

gaussian_img = gaussian(img, sigma=2, mode='constant', cval=0.0)

img2 = (img - gaussian_img)*2.

img3 = img + img2

from matplotlib import pyplot as plt
plt.imshow(img3, cmap="gray")


"""
from skimage import io
from skimage.filters import unsharp_mask

img = img_as_float(io.imread("/Users/rushirajsinhparmar/Downloads/DIP-Vectorly/globo-Scene-010.png", as_gray=True))

#Radius defines the degree of blurring
#Amount defines the multiplication factor for original - blurred image
unsharped_img = unsharp_mask(img, radius=10, amount=2)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(unsharped_img, cmap='gray')
ax2.title.set_text('Unsharped Image')

plt.show()