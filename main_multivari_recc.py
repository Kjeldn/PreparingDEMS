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
from random import randint

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
    
#%% LOAD ARRAYS, REMOVE NaN, GRAYSCALES, SAFE CANNY EDGES
for i in range(len(path)):
    file = gdal.Open(path[i])  
    band = file.GetRasterBand(1)
    exec("gt"+str(i)+" = file.GetGeoTransform()")
    exec("array"+str(i)+" = band.ReadAsArray()")
    exec("array"+str(i)+"[array"+str(i)+"==np.min(array"+str(i)+")]=0")
    
    plt.clf()
    exec("fig = plt.imshow(array"+str(i)+", cmap='gray', interpolation='nearest')")
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp"+str(i)+".png", bbox_inches='tight', pad_inches = 0)

    img = cv2.imread(wdir+"\\temp"+str(i)+".png",0)
    ht, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lt = 0.5*ht
    img[img==255]=np.mean(img)
    exec("edges"+str(i)+" = cv2.Canny(img,lt,ht)")

#%% PATCH SELECTION
w=int(min(img.shape)/10)
bol = np.zeros(img.shape)
bol[img==np.max(img)]=1
j=-1
while p <= target:    
    x_i=randint(0+w,img.shape[0]-w)                 
    y_i=randint(0+w,img.shape[1]-w)
    select = np.zeros(img.shape)
    select[x_i-w:x_i+w,y_i-w:y_i+w]=1
    if sum(sum(select*bol)) >= 1:
        continue
    else:
        j=j+1
        patch=edges0[x_i-w:x_i+w,y_i-w:y_i+w]
        
    #%% RECC WINDOW MATCHING
    sum_patch = sum(sum(patch))
    w = int(select.shape[0]/2)
    for i in range(1,len(path)):
        exec("RECC"+str(i)+" = np.zeros(img.shape)")
        for x in range(w,img.shape[0]-w):
            for y in range(w,img.shape[1]-w):
                exec("RECC"+str(i)+"[x,y] = sum(sum(np.multiply(edges"+str(i)+"[x-w:x+w,y-w:y+w],patch)))/(sum(sum(edges"+str(i)+"[x-w:x+w,y-w:y+w]))+sum_patch)")
      
    #%% CVn SCORE
    n=4
    for i in range(1,len(path)):
        exec("max_one = np.partition(RECC"+str(i)+".flatten(), -1)[-1]")
        exec("max_n   = np.partition(RECC"+str(i)+".flatten(), -(n+1))[-(n+1)]")
        exec("r_max = np.where(RECC"+str(i)+" >= max_one)[0][0]")
        exec("c_max = np.where(RECC"+str(i)+" >= max_one)[1][0]")
        exec("r_i   = np.where(RECC"+str(i)+" >= max_n)[0][0:-1]")
        exec("c_i   = np.where(RECC"+str(i)+" >= max_n)[1][0:-1]")
        exec("CV"+str(i)+" = sum(np.sqrt(np.square(r_max-r_i)+np.square(c_max-c_i)))")    
    
#%% MAX RECC SELECT
for i in range(1,len(path)):
    exec("selectmax"+str(i)+"_x = np.where(RECC"+str(i)+" >= np.max(RECC"+str(i)+"))[1][0]")
    exec("selectmax"+str(i)+"_y = np.where(RECC"+str(i)+" >= np.max(RECC"+str(i)+"))[0][0]")

#%% CONVERT TO ARRAY INDICES
for i in range(len(path)):
     exec("fact_x = array"+str(i)+".shape[0]/img.shape[0]")
     exec("fact_y = array"+str(i)+".shape[1]/img.shape[1]")
     if i == 0:
         exec("select"+str(i)+"_xa = 45*fact_x")
         exec("select"+str(i)+"_ya = 140*fact_y") 
     else:
         exec("select"+str(i)+"_xa = selectmax"+str(i)+"_x*fact_x")
         exec("select"+str(i)+"_ya = selectmax"+str(i)+"_y*fact_y")

#%% GEOREFERENCE THE POINTS
#for i in range(len(path)):
#    if i >= 1:
#        exec("coordinates = coor"+str(i))
#        os.system("gdal_translate -of GTiff -gcp " + str(coordinates[0,0]) +" "+ str(coordinates[0,1]) +" "+ str(coor0[0,0]) +" "+ str(coor0[0,1]) 
#        +" -gcp " + str(coordinates[1,0]) +" "+ str(coordinates[1,1]) +" "+ str(coor0[1,0]) +" "+ str(coor0[1,1]) 
#        +" -gcp " + str(coordinates[2,0]) +" "+ str(coordinates[2,1]) +" "+ str(coor0[2,0]) +" "+ str(coor0[2,1]) 
#        +" -gcp " + str(coordinates[3,0]) +" "+ str(coordinates[3,1]) +" "+ str(coor0[3,0]) +" "+ str(coor0[3,1]) 
#        +" \""+path[i]+"\" \""+wdir+"\\temp"+files[i]+".tif\"")
#        os.system("gdalwarp -r near -tps -co COMPRESS=NONE \""+wdir+"\\temp.tif\" \""+wdir+"\\"+files[i]+"_adjusted.tif\"")

#%% CHECK BY PLOTTING
array = edges0
plt.imshow(array, cmap='gray', interpolation='nearest')
#plt.scatter(select2_xa,select2_ya)

#%% MASK
bol = np.zeros(img.shape)
bol[img==np.max(img)]=1

