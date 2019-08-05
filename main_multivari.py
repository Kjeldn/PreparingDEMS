#%% INPUT
wdir  = r"E:\Tulips\DEM"
files = ["1","2","3"]

#%% PACKAGES AND FUNCTIONS
import rasterio
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def calculate_slope(DEM):
        gdal.DEMProcessing('slope.tif', DEM, 'slope', alg = 'ZevenbergenThorne', scale = 500000)
        with rasterio.open('slope.tif') as dataset:
            slope=dataset.read(1)
        return slope
    
def odd(f):
        return np.ceil(f / 2.) * 2 + 1
    
def raster_calc(slope,p,u):
    out = np.ones(slope.shape)
    hist = np.histogram(slope[slope>0],bins=90,density=True)
    for h in range(10,len(hist[0])):
        if hist[0][h] <= p:
            t1=h
            break
    t2=u
    for h in range(u,len(hist[0])-1):
        if hist[0][h+1] > hist[0][h]:
            t2=h
    out[slope>t2] = 0
    out[slope<t1] = 0
    return out

#%% CREATE PATH
path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
    
#%% LOAD ARRAY AND SLOPE, SLOPE MANIPULATION, GAUSSIAN FILTER, SAVING IMAGE 
for i in range(len(path)):
    file = gdal.Open(path[i])  
    band = file.GetRasterBand(1)
    exec("gt"+str(i)+" = file.GetGeoTransform()")
    exec("array"+str(i)+" = band.ReadAsArray()")
    exec("slope"+str(i)+" = calculate_slope(path[i])")
    exec("slope"+str(i)+"_p = raster_calc(slope"+str(i)+",1/2000,75)")
    exec("slope"+str(i)+"_pp = ndimage.gaussian_filter(slope"+str(i)+"_p, sigma=10, order=0)")
    
    plt.clf()
    exec("fig = plt.imshow(slope"+str(i)+"_pp, cmap='hot', interpolation='nearest')")
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp"+str(i)+".png", bbox_inches='tight', pad_inches = 0)
      
#%% HARRIS CORNER DETECTION
for i in range(len(path)): 
    exec("size_x = len(slope"+str(i)+"[:,0])")
    exec("size_y = len(slope"+str(i)+"[0,:])")    
    bsize  = int(odd(min(size_x,size_y)/200))
    ksize  = int(min(31,odd(min(size_x,size_y)/1000)))
    target = 4
    thresh = 0.01
    k      = 0.01
    for j in range(0,500):
        img = cv2.imread(wdir+"\\temp"+str(i)+".png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,bsize,ksize,k)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(20,20),(-1,-1),criteria)
        if len(corners) == target+1:
            break
        elif len(corners) <= target:
            thresh = thresh+0.01
            k=k+0.0001
        elif len(corners) >= target+2:
            thresh = thresh-0.01
            k=k+0.0001
    fact = size_x/len(img)
    file = gdal.Open(path[i])  
    gt = file.GetGeoTransform()
    corn = corners * fact
    exec("corn"+str(i)+" = corn[1:,:]")
    exec("coor"+str(i)+" = np.matrix([gt[0] + gt[1]*corn[:,0] + gt[2]*corn[:,1],gt[3] + gt[4]*corn[:,0] + gt[5]*corn[:,1]]).transpose()")     
    print('Image nr '+str(i)+' processed after '+str(j)+' alternations. Found '+str(len(corners)-1)+' corners.')

#%% GEOREFERENCE THE POINTS
for i in range(len(path)):
    if i >= 1:
        exec("coordinates = coor"+str(i))
        os.system("gdal_translate -of GTiff -gcp " + str(coordinates[0,0]) +" "+ str(coordinates[0,1]) +" "+ str(coor0[0,0]) +" "+ str(coor0[0,1]) 
        +" -gcp " + str(coordinates[1,0]) +" "+ str(coordinates[1,1]) +" "+ str(coor0[1,0]) +" "+ str(coor0[1,1]) 
        +" -gcp " + str(coordinates[2,0]) +" "+ str(coordinates[2,1]) +" "+ str(coor0[2,0]) +" "+ str(coor0[2,1]) 
        +" -gcp " + str(coordinates[3,0]) +" "+ str(coordinates[3,1]) +" "+ str(coor0[3,0]) +" "+ str(coor0[3,1]) 
        +" \""+path[i]+"\" \""+wdir+"\\temp"+files[i]+".tif\"")
        os.system("gdalwarp -r near -tps -co COMPRESS=NONE \""+wdir+"\\temp.tif\" \""+wdir+"\\"+files[i]+"_adjusted.tif\"")

#%% PLOT WHATEVER YOU WANT
check = corn2
array = array2
plt.imshow(array, cmap='hot', interpolation='nearest')
for l in range(len(check)):
    plt.scatter(check[l,0],check[l,1])
    