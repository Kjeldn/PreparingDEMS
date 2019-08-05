#%% INPUT
wdir = r"E:\Tulips\DEM"
files = ["3","4"]

ppp   = 15   # Nr. of GCPs per photo (points per photo)
ft    = 0.5  # Patch must at least contain ft times the maximum of the sum of edges found in any potential patch
md    = 25   # Max  shifting distance considered feasible
cvn   = 4    # CVn score selection based on n next largest peaks (default: n=4)
maxcv = 50   # Max CVn score considered a feasible match

#%% PACKAGES AND FUNCTIONS
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from random import randint
from scipy.ndimage.filters import convolve
from scipy import misc
from math import cos, sin, asin, sqrt, radians
import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)
    
def calc_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 1000* 6371 * c
    return m

#%% INITIALIZE SOME THINGS
path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
steps = len(path)+(len(path)-1)*ppp+len(path)-1-1
clist = list(np.random.choice(range(256), size=ppp))
    
#%% LOAD ARRAYS, REMOVE NaN, GRAYSCALES, SAFE CANNY EDGES
for i in range(len(path)):
    print("["+str(i)+"/"+str(steps)+"] Creating edges for image nr. "+str(i)+" ...")
    exec("file"+str(i)+" = gdal.Open(path[i])")  
    exec("band = file"+str(i)+".GetRasterBand(1)")
    exec("gt"+str(i)+" = file"+str(i)+".GetGeoTransform()")
    exec("array"+str(i)+" = band.ReadAsArray()")
    exec("array"+str(i)+"[array"+str(i)+"==np.min(array"+str(i)+")]=np.NaN")
    exec("under = np.percentile(array"+str(i)+"[~np.isnan(array"+str(i)+")],5)")
    exec("upper = np.percentile(array"+str(i)+"[~np.isnan(array"+str(i)+")],98.5)")
    exec("array"+str(i)+"[array"+str(i)+"<=under]=under")
    exec("array"+str(i)+"[array"+str(i)+">=upper]=upper")
        
    plt.clf()
    exec("fig = plt.imshow(array"+str(i)+", cmap='gray', interpolation='nearest')")
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp"+str(i)+".png", bbox_inches='tight', pad_inches = 0)
    img = cv2.imread(wdir+"\\temp"+str(i)+".png",0)
    exec("img"+str(i)+" = img") 
    
    ht, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lt = 0.5*ht
    mask = np.zeros(img.shape)
    mask[img==255]=1
    img[img==255]=np.mean(img)
    exec("mask"+str(i)+" = mask")
    mask_f=ndimage.gaussian_filter(mask, sigma=2, order=0)    
    exec("edges"+str(i)+" = cv2.Canny(img,lt,ht,apertureSize=3)")
    exec("edges"+str(i)+"[mask_f>=0.1]=0")
    exec("sumedges"+str(i)+" = np.zeros(edges"+str(i)+".shape)")
    w = 35#int(min(img.shape)*(pt/2))
    v = int(w/4)
    for x in range(w,img.shape[0]-w):
        for y in range(w,img.shape[1]-w):
            exec("sumedges"+str(i)+"[x,y] = sum(sum(edges"+str(i)+"[x-w:x+w,y-w:y+w]))")

#%% PATCH SELECTION
fact_x_0  = array0.shape[0]/img.shape[0]
fact_y_0  = array0.shape[1]/img.shape[1]
x_b_0 = img0.shape[0]
y_b_0 = img0.shape[1]
csv1 = np.zeros((ppp,2))
csv2 = np.zeros((ppp,2))
for i in range(1,len(path)):
    print("Selecting patches for image nr. "+str(i)+" ...")
    j=-1
    exec("fact_x = array"+str(i)+".shape[0]/img.shape[0]")
    exec("fact_y = array"+str(i)+".shape[1]/img.shape[1]")
    exec("target"+str(i)+" = []")
    exec("origin"+str(i)+" = []")
    exec("dist"+str(i)+" = np.zeros(ppp)")
    exec("target"+str(i)+"_x = []")
    exec("target"+str(i)+"_y = []")
    exec("origin"+str(i)+"_x = []")
    exec("origin"+str(i)+"_y = []")
    exec("target"+str(i)+"_la = []")
    exec("target"+str(i)+"_lo = []")
    exec("origin"+str(i)+"_pixel = []")
    exec("origin"+str(i)+"_line = []")
    while j <= ppp-2:
        x_i=randint(w,x_b_0-w)                 
        y_i=randint(w,y_b_0-w)
        target = edges0[x_i-w:x_i+w,y_i-w:y_i+w]
        exec("check1 = mask"+str(i)+"[x_i,y_i]")
        exec("check2 = sumedges"+str(i)+"[x_i,y_i]") 
        exec("mini = np.max(sumedges"+str(i)+")*ft")
        if check1 <= 0 and check2 >= mini:
            
            #%% RECC WINDOW MATCHING
            sum_target = sum(sum(target))
            exec("RECC = np.zeros(img"+str(i)+".shape)")
            exec("x_b = img"+str(i)+".shape[0]")
            exec("y_b = img"+str(i)+".shape[1]")
            for x in range(w,x_b-w):
                for y in range(w,y_b-w):
                    exec("RECC[x,y] = sum(sum(np.multiply(edges"+str(i)+"[x-w:x+w,y-w:y+w],target)))/(sum(sum(edges"+str(i)+"[x-w:x+w,y-w:y+w]))+sum_target)")
              
            #%% CVn SCORE
            max_one = np.partition(RECC.flatten(), -1)[-1]
            max_n   = np.partition(RECC.flatten(), -(cvn+1))[-(cvn+1)]
            r_max   = np.where(RECC >= max_one)[0][0]
            c_max   = np.where(RECC >= max_one)[1][0]
            r_i     = np.where(RECC >= max_n)[0][0:-1]
            c_i     = np.where(RECC >= max_n)[1][0:-1]
            CV      = sum(np.sqrt(np.square(r_max-r_i)+np.square(c_max-c_i)))
            
            #%% MAX RECC SELECT
            exec("lon = gt"+str(i)+"[0] + gt"+str(i)+"[1]*c_max*fact_x + gt"+str(i)+"[2]*r_max*fact_y")
            exec("lat = gt"+str(i)+"[3] + gt"+str(i)+"[4]*c_max*fact_x + gt"+str(i)+"[5]*r_max*fact_y")
            lon_o = gt0[0] + gt0[1]*y_i*fact_x_0 + gt0[2]*x_i*fact_y_0
            lat_o = gt0[3] + gt0[4]*y_i*fact_x_0 + gt0[5]*x_i*fact_y_0
            dist = calc_distance(lat,lon,lat_o,lon_o)
            if CV <= maxcv and dist <= md:
                csv1[j,0]=lat_o
                csv1[j,1]=lon_o
                csv2[j,0]=lat
                csv2[j,1]=lon
                j=j+1
                print("["+str(len(path)+(i-1)*ppp+j)+"/"+str(steps)+"] Matched target nr. "+str(j)+". ("+str(CV)+","+str(calc_distance(lat,lon,lat_o,lon_o))+")")
                exec("mask"+str(i)+"[x_i-v:x_i+v,y_i-v:y_i+v]=1")
                exec("target"+str(i)+".append(target)")
                exec("origin"+str(i)+".append(edges"+str(i)+"[r_max-w:r_max+w,c_max-w:c_max+w])")
                
                exec("dist"+str(i)+"[j] = dist")
                
                exec("target"+str(i)+"_x.append(x_i*fact_x_0)")
                exec("target"+str(i)+"_y.append(y_i*fact_y_0)")
                exec("origin"+str(i)+"_x.append(r_max*fact_x)")
                exec("origin"+str(i)+"_y.append(c_max*fact_y)")
                
                exec("target"+str(i)+"_la.append(lat_o)")
                exec("target"+str(i)+"_lo.append(lon_o)")
            elif CV <= maxcv:
                print("Insufficient match (dist) ...")
            elif dist <= md:
                print("Insufficient match (CV) ...")   
            else:
                print("Insufficient match (both) ...")
    exec("target"+str(i)+"_x = np.array(target"+str(i)+"_x)")                      
    exec("target"+str(i)+"_y = np.array(target"+str(i)+"_y)")
    exec("origin"+str(i)+"_x = np.array(origin"+str(i)+"_x)")
    exec("origin"+str(i)+"_y = np.array(origin"+str(i)+"_y)")

#%% CHECK BY PLOTTING
plt.imshow(array0, cmap='gray', interpolation='nearest')
plt.scatter(target1_y,target1_x,c=clist)

#%%
plt.imshow(array1, cmap='gray', interpolation='nearest')
plt.scatter(origin1_y,origin1_x,c=clist)

#%%
#plt.imshow(origin1[8], cmap='gray', interpolation='nearest')

#%% REMOVE BOXPLOT FLIERS
gcplist = []
gcplist.append(" ")
for i in range(1,len(path)):
    exec("dist"+str(i)+"_f = dist"+str(i)+".copy()")
    exec("target"+str(i)+"_x_f = target"+str(i)+"_x.copy()")
    exec("target"+str(i)+"_y_f = target"+str(i)+"_y.copy()")
    exec("origin"+str(i)+"_x_f = origin"+str(i)+"_x.copy()")
    exec("origin"+str(i)+"_y_f = origin"+str(i)+"_y.copy()")    
    exec("target"+str(i)+"_lo_f = target"+str(i)+"_lo.copy()")
    exec("target"+str(i)+"_la_f = target"+str(i)+"_la.copy()")
    fliers = np.zeros(1)
    while len(fliers) >= 1:
        exec("box = plt.boxplot(dist"+str(i)+"_f)")
        fliers = box["fliers"][0].get_data()[1]
        flier_indices = np.zeros(len(fliers))
        for j in range(len(fliers)):
            exec("flier_indices[j] = np.where(dist"+str(i)+"==fliers[j])[0][0]")
        exec("dist"+str(i)+"_f = np.delete(dist"+str(i)+"_f,flier_indices)")
        exec("target"+str(i)+"_x_f = np.delete(target"+str(i)+"_x_f,flier_indices)")
        exec("target"+str(i)+"_y_f = np.delete(target"+str(i)+"_y_f,flier_indices)")
        exec("origin"+str(i)+"_x_f = np.delete(origin"+str(i)+"_x_f,flier_indices)")
        exec("origin"+str(i)+"_y_f = np.delete(origin"+str(i)+"_y_f,flier_indices)")
        exec("target"+str(i)+"_lo_f = np.delete(target"+str(i)+"_lo_f,flier_indices)")
        exec("target"+str(i)+"_la_f = np.delete(target"+str(i)+"_la_f,flier_indices)")
    
    gcplist.append(" ")
    exec("clist"+str(i)+" = list(np.random.choice(range(256), len(dist"+str(i)+"_f)))")
    for k in range(length):
        gcplist[i] = gcplist[i]+"-gcp "+str(origin1_y_f[k])+" "+str(origin1_x_f[k])+" "+str(target1_lo_f[k])+" "+str(target1_la_f[k])+" "

#%% CHECK BY PLOTTING
plt.imshow(array0, cmap='gray', interpolation='nearest')
plt.scatter(target1_y_f,target1_x_f,c=clist1)

#%%
plt.imshow(array1, cmap='gray', interpolation='nearest')
plt.scatter(origin1_y_f,origin1_x_f,c=clist1)

#%% GEOREFERENCE USING FOUND POINTS
for i in range(1,len(path)):
    print("["+str(len(path)+(len(path)-1)*ppp-1+i)+"/"+str(steps)+"] Georeferencing image nr. "+str(i))
    path1 = wdir+"\\temp"+files[i]+".tif"
    path2 = wdir+"\\"+files[i]+"_adjusted.tif"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist[i]+"\""+path[i]+"\" \""+path1+"\"")
    os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")
