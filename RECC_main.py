#%% INPUT
#wdir = r"E:\Tulips\DEM"
#files = ["0","1","2"]

wdir = r"E:\190806_DEMs_Boomgaard"
files = ["c03_termote-De Boomgaard-201906100000_DEM","c03_termote-De Boomgaard-201906250000_DEM","c03_termote-De Boomgaard-201907110000_DEM"]


input_type = 0    # [0: DEM] [1: Slope of DEM] [2: Orthomosaic RGB]
q_l = 0.1         # [R] lower quantile of the array to be capped      
q_h = 99.99       # [R] higher quantile of the array to be capped
aperture_size = 3 # [3: DEM] [7: Ortho]
outlier_type = 0  # [0: Boxplot fliers] [1: 99% Confidence interval]
ppp = 25          # Points per photo
ft = 0.7          # [R] feature score threshold
cv_num = 4        # [R] Consider ... peaks in RECC window matching
cv_max = 40       # Approve matches only if CV score is under ...
dst_max = 20      # Approve matches only if distance is under ...

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
from scipy.ndimage.filters import convolve
from scipy import misc
from math import cos, sin, asin, sqrt, radians
import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def calculate_slope(DEM):
        gdal.DEMProcessing('slope.tif', DEM, 'slope', alg = 'ZevenbergenThorne', scale = 500000)
        with rasterio.open('slope.tif') as dataset:
            slope=dataset.read(1)
        return slope
    
def calc_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 1000 * 6371 * c
    return m

#%% INITIALIZE
path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
steps = (1+(len(path)-1)*(ppp+3))/100

#%% TARGET IMAGE
print("[0%] Getting edges for target image.")
file = gdal.Open(path[0])
gt_0 = file.GetGeoTransform()
if input_type == 0:
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    array[array==np.min(array)]=np.NaN
    under = np.percentile(array[~np.isnan(array)],q_l)
    upper = np.percentile(array[~np.isnan(array)],q_h)
    array[array<=under]=under
    array[array>=upper]=upper
elif input_type == 1:
    slope = calculate_slope(file)
    slope[slope<= 0] = np.NaN 
    array = slope
elif input_type == 2:
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()

plt.clf()
fig = plt.imshow(array, cmap='gray', interpolation='nearest')
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig(wdir+"\\temp.png", bbox_inches='tight', pad_inches = 0)
img = cv2.imread(wdir+"\\temp.png",0)
mask_0 = np.zeros(img.shape)
mask_0[img==255]=1
mask_0_f=ndimage.gaussian_filter(mask_0, sigma=2, order=0)  
fact_x_0 = array.shape[0]/img.shape[0]
fact_y_0 = array.shape[1]/img.shape[1]
x_b_0 = img.shape[0]
y_b_0 = img.shape[1]
w = 70
#w = int(770/max(fact_x_0,fact_y_0))
v = int(w/4)

img[img>=225]=0  
img_blur = cv2.GaussianBlur(img,(5,5),0)
img_agt = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
ht, thresh_im = cv2.threshold(img_agt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lt = 0.5*ht
edges_0 = cv2.Canny(img_agt,lt,ht,aperture_size)
edges_0[mask_0_f>=0.1]=0

sumedges_0 = np.zeros(edges_0.shape)
for x in range(w,img.shape[0]-w):
    for y in range(w,img.shape[1]-w):
        sumedges_0[x,y] = sum(sum(edges_0[x-w:x+w,y-w:y+w]))
minfeatures = np.max(sumedges_0)*ft

#%%
for i in range(1,len(path)):
    print("["+"{:.0f}".format((1+(i-1)*(ppp+3))/steps)+"%] Getting edges for image nr "+str(i)+".")
    file = gdal.Open(path[i])
    gt = file.GetGeoTransform() 
    if input_type == 0:   
        band = file.GetRasterBand(1)
        array = band.ReadAsArray()
        array[array==np.min(array)]=np.NaN
        under = np.percentile(array[~np.isnan(array)],q_l)
        upper = np.percentile(array[~np.isnan(array)],q_h)
        array[array<=under]=under
        array[array>=upper]=upper
    elif input_type == 1:
        slope = calculate_slope(file)
        slope[slope<= 0] = np.NaN 
        array = slope     
    plt.clf()
    fig = plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp.png", bbox_inches='tight', pad_inches = 0)
    img = cv2.imread(wdir+"\\temp.png",0)
    mask = np.zeros(img.shape)
    mask[img==255]=1
    mask_f=ndimage.gaussian_filter(mask, sigma=2, order=0)  
    
    img[img>=225]=0  
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    img_agt = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    ht, thresh_im = cv2.threshold(img_agt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lt = 0.5*ht
    edges = cv2.Canny(img_agt,lt,ht,aperture_size)
    edges[mask_f>=0.1]=0
            
    print("["+"{:.0f}".format((2+(i-1)*(ppp+3))/steps)+"%] Matching patches for image nr "+str(i)+".")
    mask_0_c = mask_0.copy()
    fact_x = array.shape[0]/img.shape[0]
    fact_y = array.shape[1]/img.shape[1]
    x_b = img.shape[0]
    y_b = img.shape[1]
    RECC = np.zeros(img.shape)
    dist       = np.zeros(ppp)
    dist_lon   = np.zeros(ppp)
    dist_lat   = np.zeros(ppp)
    origin_x   = np.zeros(ppp)
    origin_y   = np.zeros(ppp)
    target_lon = np.zeros(ppp)
    target_lat = np.zeros(ppp)
    j=-1
    it = 0
    while j <= ppp-2 and it <= 10*ppp:
        x_i_0 = randint(w,x_b_0-w)
        y_i_0 = randint(w,y_b_0-w)
        check1 = mask_0_c[x_i_0,y_i_0]
        check2 = sumedges_0[x_i_0,y_i_0]
        if check1 <= 0 and check2 >= minfeatures:
            it = it+1
            target = edges_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
            sum_target = sum(sum(target))
            for x in range(w,x_b-w):
                for y in range(w,y_b-w):
                    RECC[x,y]=sum(sum(np.multiply(edges[x-w:x+w,y-w:y+w],target)))/(sum(sum(edges[x-w:x+w,y-w:y+w]))+sum_target)           
            max_one  = np.partition(RECC.flatten(),-1)[-1]
            max_n    = np.partition(RECC.flatten(),-cv_num-1)[-cv_num-1]
            x_i    = np.where(RECC >= max_one)[1][0]
            y_i    = np.where(RECC >= max_one)[0][0]
            x_n      = np.where(RECC >= max_n)[1][0:-1]
            y_n      = np.where(RECC >= max_n)[0][0:-1]
            cv_score = sum(np.sqrt(np.square(x_i-x_n)+np.square(y_i-y_n)))          
            if cv_score <= cv_max:
                lon = gt[0] + gt[1]*x_i*fact_x + gt[2]*y_i*fact_y
                lat = gt[3] + gt[4]*x_i*fact_x + gt[5]*y_i*fact_y
                lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_x_0 + gt_0[2]*x_i_0*fact_y_0
                lat_0 = gt_0[3] + gt_0[4]*y_i_0*fact_x_0 + gt_0[5]*x_i_0*fact_y_0
                dst = calc_distance(lat,lon,lat_0,lon_0)
                if dst <= dst_max:
                    j=j+1
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Succesful patch-match nr "+str(j+1)+" of "+str(ppp)+".")
                    mask_0_c[x_i_0-v:x_i_0+v,y_i_0-v:y_i_0+v]=1
                    dist[j]       = dst
                    dist_lon[j]    = lon_0-lon
                    dist_lat[j]    = lat_0-lat
                    origin_x[j]   = x_i*fact_x
                    origin_y[j]   = y_i*fact_y
                    target_lon[j] = lon_0
                    target_lat[j] = lat_0
                else:
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Match failed.")
            else:
                print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+",-) Match failed.")
            if it == 10*ppp+1:
                dist = dist[dist!=0]
                dist_lon = dist_lon[dist_lon!=0]
                dist_lat = dist_lat[dist_lat!=0]
                origin_x = origin_x[origin_x!=0]
                origin_y = origin_y[origin_y!=0]
                target_lon = target_lon[target_lon!=0]
                target_lat = target_lat[target_lat!=0]
    
    gcplist = " "
    size = len(dist)
    if outlier_type == 0:
        fliers = np.zeros(1)
        while len(fliers) >= 1:
            box = plt.boxplot(dist)
            fliers = box["fliers"][0].get_data()[1]
            flier_indices = np.zeros(len(fliers))
            for j in range(len(fliers)):
                flier_indices[j] = np.where(dist==fliers[j])[0][0]
            flier_indices = flier_indices.astype(int)
            dist       = np.delete(dist,flier_indices)
            dist_lon   = np.delete(dist_lon,flier_indices)
            dist_lat   = np.delete(dist_lat,flier_indices)
            origin_x   = np.delete(origin_x,flier_indices)
            origin_y   = np.delete(origin_y,flier_indices)
            target_lon = np.delete(target_lon,flier_indices)
            target_lat = np.delete(target_lat,flier_indices)
        print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] {BPFR} Removed "+str(size-len(dist))+" outliers.")   
    elif outlier_type == 1:
        mean = np.mean(dist)
        std = np.std(dist)
        bol1 = np.linspace(0,len(dist)-1,len(dist))
        bol2 = np.zeros(dist.shape)
        bol2[dist<=mean-2.58*(std/sqrt(len(dist)))]=1
        bol2[dist>=mean+2.58*(std/sqrt(len(dist)))]=1
        bol = (bol1*bol2).astype(int)
        dist = np.delete(dist,bol)
        dist_lon   = np.delete(dist_lon,bol)
        dist_lat   = np.delete(dist_lat,bol)
        origin_x   = np.delete(origin_x,bol)
        origin_y   = np.delete(origin_y,bol)
        target_lon = np.delete(target_lon,bol)
        target_lat = np.delete(target_lat,bol)
        print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] {CIR} Removed "+str(size-len(dist))+" outliers.")
    
    box_lon = plt.boxplot(dist_lon)
    fliers_lon = box_lon["fliers"][0].get_data()[1]
    flier_indices_lon = np.zeros(len(fliers_lon))
    for j in range(len(fliers_lon)):
        flier_indices_lon[j] = np.where(dist_lon==fliers_lon[j])[0][0]
    box_lat = plt.boxplot(dist_lat)
    fliers_lat = box_lat["fliers"][0].get_data()[1]
    flier_indices_lat = np.zeros(len(fliers_lat))
    for j in range(len(fliers_lat)):
        flier_indices_lat[j] = np.where(dist_lat==fliers_lat[j])[0][0]
    flier_indices_lonlat = (np.intersect1d(flier_indices_lon,flier_indices_lat)).astype(int)    
    size = len(flier_indices_lonlat)
    dist = np.delete(dist,flier_indices_lonlat)
    origin_x   = np.delete(origin_x,flier_indices_lonlat)
    origin_y   = np.delete(origin_y,flier_indices_lonlat)
    target_lon = np.delete(target_lon,flier_indices_lonlat)
    target_lat = np.delete(target_lat,flier_indices_lonlat)
    print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] {BPFR} Removed "+str(size)+" lon-lat outliers.")
    
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_x[k])+" "+str(origin_y[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" " 
    
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Georeferencing image nr "+str(i)+".")
    path1 = wdir+"\\temp.tif"
    path2 = wdir+"\\"+files[i]+"_adjusted.tif"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path[i]+"\" \""+path1+"\"")
    os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    

print("[100%] Done.")

#%%
#plt.imshow(array, cmap='gray', interpolation='nearest')
#plt.scatter(origin_x,origin_y)

#%%
#plt.imshow(sumedges_0>=minfeatures)


