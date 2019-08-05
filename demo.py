import rasterio
from osgeo import gdal
import cv2
import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os

#%% DEFINITIONS
def calculate_slope(DEM):
        gdal.DEMProcessing('slope.tif', DEM, 'slope', alg = 'ZevenbergenThorne', scale = 500000)
        with rasterio.open('slope.tif') as dataset:
            slope=dataset.read(1)
        return slope

#%% LOADING DATA
file = gdal.Open(r"E:\190723_Demo_Hendrik_de_Heer\DEM\20190603_modified.tif")   
    
band = file.GetRasterBand(1)
gt = file.GetGeoTransform()
array = band.ReadAsArray()
    
plt.imshow(array, cmap='hot', interpolation='nearest')

#%% ZEVENBERGEN-THORNE SLOPE
slope = calculate_slope(file)

plt.imshow(slope, cmap='hot', interpolation='nearest')

#%% QGIS RASTER CALCULATOR   
hist = np.histogram(slope[slope>0],bins=90,density=True)
for h in range(10,len(hist[0])):
    if hist[0][h] <= 1/1000:
        thresh1=h
        break
for h in range(thresh1,len(hist[0])):
    if hist[0][h+1] > hist[0][h]:
        thresh2=h
        break
        
#%%
slope[slope<0]  = 0
slope[slope>thresh2] = 0
slope[slope<thresh1] = 0
slope[slope>thresh1] = 1

#%% SMOOTHING
slope_p = ndimage.gaussian_filter(slope, sigma=10, order=0)

plt.imshow(slope_p, cmap='hot', interpolation='nearest')
    
#%% EXTRACT FIGURE
fig = plt.imshow(slope_p, cmap='hot', interpolation='nearest')
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("temp.png", bbox_inches='tight', pad_inches = 0)
    
#%% HARRIS CORNER DETECTION (1/2)  
img = cv2.imread('temp.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,31,9,0.01)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(40,40),(-1,-1),criteria)
        
#%% HARRIS CORNER DETECTION (2/2)
cv2.imshow("img", img); 
for i in range(1, len(corners)):
    cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)
cv2.waitKey(0); cv2.destroyAllWindows()
    
#%% TRANSLATE CORNERS TO ARRAY INDICES
corn = corners[1:,:]
fact = len(array)/len(img)
corn = corn*fact
plt.imshow(array, cmap='hot', interpolation='nearest')
for l in range(len(corn)):
    plt.scatter(corn[l,0],corn[l,1])
        
#%% TRANSLATE CORNERS TO COORDINATES
X = gt[0] + gt[1]*corn[:,0] + gt[2]*corn[:,1]
Y = gt[3] + gt[4]*corn[:,0] + gt[5]*corn[:,1]
coordinates = np.matrix([X,Y]).transpose()
truth = coordinates+0.0001

#%% GEOREFERENCE
os.system("gdal_translate -of GTiff -gcp " + str(coordinates[0,0]) +" "+ str(coordinates[0,1]) +" "+ str(truth[0,0]) +" "+ str(truth[0,1]) + 
          " -gcp " + str(coordinates[1,0]) +" "+ str(coordinates[1,1]) +" "+ str(truth[1,0]) +" "+ str(truth[1,1]) + 
          " -gcp " + str(coordinates[2,0]) +" "+ str(coordinates[2,1]) +" "+ str(truth[2,0]) +" "+ str(truth[2,1]) +
          " -gcp " + str(coordinates[3,0]) +" "+ str(coordinates[3,1]) +" "+ str(truth[3,0]) +" "+ str(truth[3,1]) +
          " \"C:/Users/Martijn/AppData/Local/Temp/20190603_modified.tif\" \"E:/temp.tif\"")
os.system("gdalwarp -r near -tps -co COMPRESS=NONE \"E:/temp.tif\" \"E:/file0.tif\"")




