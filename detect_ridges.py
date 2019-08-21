import numpy as np
import cv2
import gdal
import matplotlib.pyplot as plt

def get_ridges_array(dem):
    dem[dem == np.amin(dem)] = 0
    dem[dem > 10] = 0
    
    kernel = np.ones((8,1),np.float32)/8
    filtered_dem = cv2.filter2D(dem,-1,kernel)
    for i in range(2):
        filtered_dem = cv2.filter2D(filtered_dem,-1,kernel)
    
    n=25
    kernel = np.ones((n,n),np.float32)/(n**2)
    smooth = cv2.filter2D(dem,-1,kernel)
    
    ridges = (filtered_dem-smooth)
    ridges[ridges > 0] = 1
    ridges = ridges.astype(np.uint8)
    
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
    
#write_ridges_array(r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\c01_verdonk-Wever oost-201908041528_DEM-GR.tif", r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\c01_verdonk-Wever oost-201908041528_DEM-GR_ridges.tif")
