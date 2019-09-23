import numpy as np
import cv2
import gdal
import matplotlib.pyplot as plt

def get_ridges_array(dem):
    mask = np.zeros(dem.shape)
    if np.sum(dem==0) > np.sum(dem==np.min(dem)):
        mask[dem == 0]               = 1
    else:
        mask[dem == np.min(dem)]   = 1
    dem[dem == np.amin(dem)] = 0
    dem[dem > 10] = 0
    
    dem_f = cv2.GaussianBlur(dem,(11,11),0)
    smooth = cv2.GaussianBlur(dem_f,(15,15),0)
    ridges = (dem_f-smooth)
    #kernel = np.ones((n,n),np.float32)/(n**2)
    #smooth = cv2.filter2D(dem_f,-1,kernel)
    mask_b = cv2.GaussianBlur(mask,(51,51),0)  
    ridges[mask>10**-10]=0  
    temp1 = np.zeros(ridges.shape)
    temp2 = np.zeros(ridges.shape)
    temp1[ridges<-0.00]=1
    temp2[ridges>-0.11]=1
    ridges = (temp1*temp2).astype(np.uint8) 
    
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
if __name__ == "__main__":
    file = gdal.Open(R"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants compare3\c07_hollandbean-Joke Visser-201906031020_DEM.tif")
    projection = file.GetProjection()
    band = file.GetRasterBand(1)
    array = get_ridges_array(band.ReadAsArray())
    file = None