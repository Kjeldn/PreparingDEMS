import rasterio
import gdal
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
    
def odd(f):
        return np.ceil(f / 2.) * 2 + 1

#%% INPUT
wdir  = r"E:\Tulips\DEM"
files = ["1","2","3"]

#%% LOADING DATA
path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
    
for i in range(len(path)):    
    file = gdal.Open(path[i])   
    
    band = file.GetRasterBand(1)
    gt = file.GetGeoTransform()
    array = band.ReadAsArray()
    
    #%% ZEVENBERGEN-THORNE SLOPE
    slope = calculate_slope(file)
    
    #%% QGIS RASTER CALCULATOR    
    hist = np.histogram(slope[slope>0],bins=90,density=True)
    for h in range(10,len(hist[0])):
        if hist[0][h] <= 1/5000:
            thresh1=h
            break
    for h in range(thresh1,len(hist[0])):
        if hist[0][h+1] > hist[0][h]:
            thresh2=h
            break
    
    slope_p=slope    
    slope_p[slope_p<0]  = 0
    slope_p[slope_p>thresh2] = 0
    slope_p[slope_p<thresh1] = 0
    slope_p[slope_p>thresh1] = 1
    
    #%% SMOOTHING
    slope_pp = ndimage.gaussian_filter(slope_p, sigma=10, order=0)
    
    #%% EXTRACT FIGURE
    plt.clf()
    fig = plt.imshow(slope_pp, cmap='hot', interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp"+str(i)+".png", bbox_inches='tight', pad_inches = 0)
        
    #%% HARRIS CORNER DETECTION (1/2)      
    bsize = int(odd(min(len(array[:,0]),len(array[0,:]))/200))
    ksize = int(min(31,odd(min(len(array[:,0]),len(array[0,:]))/1000)))
    target=4
    thresh = 0.01
    k = 0.01
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
        
    print('Image nr '+str(i+1)+' processed after '+str(j)+' alternations. Found '+str(len(corners)-1)+' corners.')
    
    #%% TRANSLATE CORNERS TO ARRAY INDICES
    corn = corners[1:,:]
    fact = len(array)/len(img)
    corn = corn*fact
    exec('corn'+str(i)+' = corn')
        
    #%% TRANSLATE CORNERS TO COORDINATES
    X = gt[0] + gt[1]*corn[:,0] + gt[2]*corn[:,1]
    Y = gt[3] + gt[4]*corn[:,0] + gt[5]*corn[:,1]
    if i == 0:
        truth = np.matrix([X,Y]).transpose()
    else:
        coordinates = np.matrix([X,Y]).transpose()

    #%% GEOREFERENCE
    if i >= 1:
        os.system("gdal_translate -of GTiff -gcp " + str(coordinates[0,0]) +" "+ str(coordinates[0,1]) +" "+ str(truth[0,0]) +" "+ str(truth[0,1]) 
        +" -gcp " + str(coordinates[1,0]) +" "+ str(coordinates[1,1]) +" "+ str(truth[1,0]) +" "+ str(truth[1,1]) 
        +" -gcp " + str(coordinates[2,0]) +" "+ str(coordinates[2,1]) +" "+ str(truth[2,0]) +" "+ str(truth[2,1]) 
        +" -gcp " + str(coordinates[3,0]) +" "+ str(coordinates[3,1]) +" "+ str(truth[3,0]) +" "+ str(truth[3,1]) 
        +" \""+path[i]+"\" \""+wdir+"\\temp"+files[i]+".tif\"")
        os.system("gdalwarp -r near -tps -co COMPRESS=NONE \""+wdir+"\\temp.tif\" \""+wdir+"\\"+files[i]+"_adjusted.tif\"")

#%%
check = corn1
plt.imshow(array, cmap='hot', interpolation='nearest')
for l in range(len(check)):
    plt.scatter(check[l,0],check[l,1])
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
#%% THINNING TEST
def nombre_transitions (image, i, j):
    transitions = 0
    if image[i - 1][j] == 0 and image[i - 1][j + 1] == 1:
        transitions += 1
    if image[i - 1][j + 1] == 0 and image[i][j + 1] == 1:
        transitions += 1
    if image[i + 1][j + 1] == 0 and image[i + 1][j] == 1:
        transitions += 1
    if image[i + 1][j] == 0 and image[i + 1][j - 1] == 1:
        transitions += 1
    if image[i + 1][j - 1] == 0 and image[i][j - 1] == 1:
        transitions += 1
    if image[i][j - 1] == 0 and image[i - 1][j - 1] == 1:
        transitions += 1
    if image[i - 1][j - 1] == 0 and image[i - 1][j] == 1:
        transitions += 1
    return transitions

def nombre_voisins_8 (image, i, j):
    #on suppose que le point est noir et n'est pas sur un bord
    voisins = 0
    for k in range(i - 1, i + 2):
        for l in range(j - 1, j + 2):
            voisins += image[k][l]
    return voisins - 1

def zhang_suen (argument_image):
    image = []
    for i in range(len(argument_image)):
        image.append([x for x in argument_image[i]])
    continuer = True
    while continuer:
        continuer = False
        a_supprimer = []
        for i in range(1, len(image) - 1):
            for j in range(1, len(image[0]) - 1):
                condition = True
                condition = condition and image[i][j] == 1
                condition = condition and 2 <= nombre_voisins_8(image, i, j) <= 6
                condition = condition and nombre_transitions(image, i, j) == 1
                condition = condition and (image[i - 1][j] == 0 or image[i][j + 1] == 0 or image[i + 1][j] == 0)
                condition = condition and (image[i][j - 1] == 0 or image[i][j + 1] == 0 or image[i + 1][j] == 0)
                if condition:
                    a_supprimer.append((i, j))
                    continuer = True
        for x in a_supprimer:
            i, j = x
            image[i][j] = 0
        for i in range(1, len(image) - 1):
            for j in range(1, len(image[0]) - 1):
                condition = True
                condition = condition and image[i][j] == 1
                condition = condition and 2 <= nombre_voisins_8(image, i, j) <= 6
                condition = condition and nombre_transitions(image, i, j) == 1
                condition = condition and (image[i - 1][j] == 0 or image[i][j + 1] == 0 or image[i][j - 1] == 0)
                condition = condition and (image[i][j - 1] == 0 or image[i - 1][j] == 0 or image[i + 1][j] == 0)
                if condition:
                    a_supprimer.append((i, j))
                    continuer = True
        for x in a_supprimer:
            i, j = x
            image[i][j] = 0
        a_supprimer = []
    return image

#%%
test = zhang_suen(wdir+"\\temp0.png")