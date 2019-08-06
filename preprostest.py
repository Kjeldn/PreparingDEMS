wdir = r"E:\Tulips\Ortho"
files = ["0","1","2"]
import rasterio
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from random import randint
from scipy.ndimage.filters import convolve
from scipy import misc
from math import cos, sin, asin, sqrt, radians
import warnings


path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
#%%
file = gdal.Open(path[0])
gt_0 = file.GetGeoTransform()

band = file.GetRasterBand(1)
array = band.ReadAsArray()
aperture_size = 3
#%%
array_0 = array
    
plt.clf()
fig = plt.imshow(array, cmap='gray', interpolation='nearest')
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig(wdir+"\\temp.png", bbox_inches='tight', pad_inches = 0)
img = cv2.imread(wdir+"\\temp.png",0)

#%%
mask_0 = np.zeros(img.shape)
mask_0[img==255]=1
mask_0_f=ndimage.gaussian_filter(mask_0, sigma=2, order=0)  
fact_x_0 = array.shape[0]/img.shape[0]
fact_y_0 = array.shape[1]/img.shape[1]
x_b_0 = img.shape[0]
y_b_0 = img.shape[1]

#%%

img_0 = img

#%%
blur = cv2.GaussianBlur(img,(5,5),0)
ht, thresh_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lt = 0.5*ht
edges_0 = cv2.Canny(img,lt,ht,aperture_size)
edges_0[mask_0_f>=0.1]=0
