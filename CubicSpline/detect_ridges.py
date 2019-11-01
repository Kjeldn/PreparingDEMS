import numpy as np
import cv2
import gdal
import matplotlib.pyplot as plt

def get_ridges_array(dem, thr):
    mask = np.zeros(dem.shape)
    if np.sum(dem==0) > np.sum(dem==np.min(dem)):
        mask[dem == 0] = 1
    else:
        mask[dem == np.min(dem)] = 1
    dem[dem == np.amin(dem)] = 0
    dem[dem > 10] = 0
    
    dem_f = cv2.GaussianBlur(dem,(11,11),0)
    smooth = cv2.GaussianBlur(dem_f,(15,15),0)
    ridges = dem_f - smooth
    ridges[mask>10**-10]=0
    temp2 = np.zeros(ridges.shape)
    temp2[ridges<thr]=1
    ridges = (temp2).astype(np.uint8) 
    
    return ridges

def write_ridges_array(src_path, dst_path):
    file = gdal.Open(src_path)
    gt = file.GetGeoTransform()
    projection = file.GetProjection()
    band = file.GetRasterBand(1)
    array = get_ridges_array(band.ReadAsArray())
    file = None
    
    driver = gdal.GetDriverByName('GTiff')
    tiff = driver.Create(dst_path, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    tiff.SetGeoTransform(gt)
    tiff.SetProjection(projection)
    tiff.GetRasterBand(1).WriteArray(array)
    tiff.GetRasterBand(1).FlushCache()
    tiff = None