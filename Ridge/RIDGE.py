#RIDGE
import numpy as np
import METAA
import gdal
import cv2
import matplotlib.pyplot as plt


path = r"\\STAMPERTJE\Data\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\ORTHODUMP\Hollandbean - Jos Schelling\2_DEM.tif"

def DemOpening(psF,path):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    dem_o                              = file.GetRasterBand(1).ReadAsArray()
    x_s, y_s                           = METAA.calc_pixsize(dem_o,gt)
    mask                               = np.zeros(dem_o.shape)
    mask[dem_o==np.min(dem_o)]         = 1
    dem                                = cv2.resize(dem_o,(int(dem_o.shape[1]*(y_s/psF)), int(dem_o.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA)
    mask                               = cv2.resize(mask,(int(mask.shape[1]*(y_s/psF)), int(mask.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA)
    fx                                 = dem_o.shape[0]/dem.shape[0]
    fy                                 = dem_o.shape[1]/dem.shape[1]
    dem[dem == np.amin(dem)] = 0
    dem[dem > 10] = 0  
    kernel = np.ones((8,1),np.float32)/8
    filtered_dem = cv2.filter2D(dem,-1,kernel)
    for i in range(2):
        filtered_dem = cv2.filter2D(filtered_dem,-1,kernel)  
    n=15
    kernel = np.ones((n,n),np.float32)/(n**2)
    smooth = cv2.filter2D(dem,-1,kernel)
    ridges = (filtered_dem-smooth)
    mask_b = cv2.GaussianBlur(mask,(51,51),0)  
    ridges[mask_b>10**-10]=0
    
    temp1 = np.zeros(ridges.shape)
    temp2 = np.zeros(ridges.shape)
    temp1[ridges<-0.01]=1
    temp2[ridges>-0.11]=1
    temp = (temp1*temp2).astype(np.uint8)    
    return temp, gt, fx, fy, mask
