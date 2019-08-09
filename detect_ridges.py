import numpy as np
import cv2
import gdal

def get_ridges_array(path):
    ds = gdal.Open(path)
    dem = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    dem[dem == np.amin(dem)] = 0
    dem[dem > 10] = 0
    
    kernel = np.ones((8,1),np.float32)/8
    filtered_dem = cv2.filter2D(dem,-1,kernel)
    for i in range(2):
        filtered_dem = cv2.filter2D(filtered_dem,-1,kernel)
    
    n=35
    kernel = np.ones((n,n),np.float32)/(n**2)
    smooth = cv2.filter2D(dem,-1,kernel)
    
    ridges = (filtered_dem-smooth)
    ridges[ridges > -0.05] = 255
    ridges = ridges.astype(np.uint8)
    
    return ridges, gt, projection

def write_ridges_array(src_path, dst_path):
    
    array, gt, projection = get_ridges_array(src_path)
    
    driver = gdal.GetDriverByName('GTiff')
    tiff = driver.Create(dst_path, array.shape[1], array.shape[0], 1, gdal.GDT_Int16)
    tiff.SetGeoTransform(gt)
    tiff.SetProjection(projection)
    tiff.GetRasterBand(1).WriteArray(array)
    tiff.GetRasterBand(1).FlushCache()
    tiff = None