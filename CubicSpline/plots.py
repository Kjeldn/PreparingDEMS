# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:33:51 2019

@author: wytze
"""
import gdal
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import Affine as A

wd = r"C:/Users/wytze/OneDrive/Documents/vanBoven/Broccoli"
path = "c01_verdonk-Wever oost-201907170731_DEM-GR"
path_ahn = "m_19fn2.tif"

if path_ahn:
    orig = gdal.Open(wd + "/" + path + ".tif")
    dst_shape = (orig.GetRasterBand(1).YSize, orig.GetRasterBand(1).XSize)
    dst_transform = A(orig.GetGeoTransform()[1], 0, orig.GetGeoTransform()[0], 0, orig.GetGeoTransform()[5],  orig.GetGeoTransform()[3])
    ahn_array = np.zeros(dst_shape)
    dst_crs = "EPSG:4326"
    
    with rasterio.open(wd + "/" + path_ahn) as src:
        source = src.read(1)
        
        with rasterio.Env():
            reproject(
                    source,
                    ahn_array,
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = dst_transform,
                    dst_crs = dst_crs,
                    respampling = Resampling.nearest
                    )
            
    source = None
    
ahn_array[ahn_array > 10] = 0

plt.imshow(ahn_array, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()