wdir = r"E:\DEMDUMP"
#files = ["0","1","2","3","4","5","6","HDH","B0","B1","B2"]
#files = ["0","2","HDH","B0","B2"]  # Varying selection {Clean, Heap, Varying NaN, Clean, Too clean}
files = ["0","2","HDH","B0","B2","5"]  # Varying selection {Clean, Heap, Varying NaN, Clean, Too clean, Large + Heap?}

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

#%%
path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")

#%%
#plt.imshow(edges0,cmap='gray')
    
#%% STANDARD [2019-08-06]
for i in range(len(path)):
    print(i)
    file = gdal.Open(path[i])
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    exec("array"+str(i)+" = array")
    array[array<=0.8*np.nanmin(array)]=np.NaN
    under = np.percentile(array[~np.isnan(array)],1)
    upper = np.percentile(array[~np.isnan(array)],99)
    array[array<=under]=under
    array[array>=upper]=upper
    exec("array"+str(i)+"_p = array")
        
    plt.clf()
    fig = plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(wdir+"\\temp.png", bbox_inches='tight', pad_inches = 0)
    
    img = cv2.imread(wdir+"\\temp.png",0)
    exec("img"+str(i)+" = img")
    mask = np.zeros(img.shape)
    mask[img==255]=1
    mask_f=ndimage.gaussian_filter(mask, sigma=2, order=0)  
    
    exec("img"+str(i)+"[img"+str(i)+">=225]=0")
    exec("img"+str(i)+"_b = cv2.GaussianBlur(img"+str(i)+",(5,5),0)")
    exec("img"+str(i)+"_ba = cv2.adaptiveThreshold(img"+str(i)+"_b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)")
    exec("ht"+str(i)+", thresh_im = cv2.threshold(img"+str(i)+"_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
    exec("lt"+str(i)+" = 0.5*ht"+str(i))
    exec("edges"+str(i)+" = cv2.Canny(img"+str(i)+"_ba,lt"+str(i)+",ht"+str(i)+",3)")
    exec("edges"+str(i)+"[mask_f>=0.1]=0")
    
#%% ADAPTIVE THRESHOLDING [2019-08-07]
s = 5
for i in range(len(path)):
    print(i)
    file    = gdal.Open(path[i])
    band    = file.GetRasterBand(1)
    array   = band.ReadAsArray()
    
    # Scale:
    arr_s                             = cv2.resize(array,(int(array.shape[1]*0.07), int(array.shape[0]*0.07)),interpolation = cv2.INTER_AREA)
    
    # Cut:
    mask                              = np.zeros(arr_s.shape)
    mask[arr_s<=0.8*np.nanmin(arr_s)] = 1
    mask_b                            = cv2.GaussianBlur(mask,(s,s),0)
    mask_hb                           = cv2.GaussianBlur(mask,(21,21),0) 
    arr_sc                            = arr_s.copy()
    arr_sc[mask_b>=10**-10]           = np.NaN
    while np.nanmax(arr_sc)-np.nanmin(arr_sc) >= 100:
        s = s+2
        mask_b                        = cv2.GaussianBlur(mask,(s,s),0)
        arr_sc[mask_b>=10**-10]       = np.NaN
    mask_hb                           = cv2.GaussianBlur(mask,(s+6,s+6),0) 
    
    # Cap:
    sort                  = np.unique(arr_sc[~np.isnan(arr_sc)])
    sortofmedian          = sort[int(len(sort)/2)]
    std                   = np.std(arr_s[~np.isnan(arr_sc)])
    cap                   = sortofmedian + 1.5*std
    arr_scc               = arr_sc.copy()
    arr_scc[arr_scc>=cap] = cap
    
    # Grayscale
    arr_sccg = arr_scc.copy()
    un       = np.nanmin(arr_sccg)
    arr_sccg = arr_sccg-un
    up       = np.nanmax(arr_sccg)
    arr_sccg = (arr_sccg/up)*255
    
    # Blur:
    arr_sccgb                        = cv2.GaussianBlur(arr_sccg,(3,3),0)
    
    # Fill NaN
    arr_sccgbf                       = arr_sccgb.copy()
    arr_sccgbf[np.isnan(arr_sccgbf)] = np.nanmean(arr_sccgbf)
    arr_sccgbf                       = arr_sccgbf.astype(np.uint8)    
    
    # Adaptive threshold:
    arr_sccgbfa                      = cv2.adaptiveThreshold(arr_sccgbf,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
    arr_sccgbfa[mask_hb>=10**-10]    = 0
    arr_sccgbfa                      = cv2.medianBlur(arr_sccgbfa,5)
    exec("save"+str(i)+" = arr_sccgbfa")

#%% ADAPTIVE THRESHOLDING (COMPACT) [2019-08-07] IMAGE SCALING UPDATE
s = 5
for i in range(len(path)):
    print(i)
    file                              = gdal.Open(path[i])
    band                              = file.GetRasterBand(1)
    gt                                = file.GetGeoTransform()
    array                             = band.ReadAsArray()
    x_s, y_s                          = pix_size(array,gt)
    arr_s                             = cv2.resize(array,(int(array.shape[1]*(y_s/0.5)), int(array.shape[0]*(x_s/0.5))),interpolation = cv2.INTER_AREA)
    mask                              = np.zeros(arr_s.shape)
    mask[arr_s<=0.8*np.nanmin(arr_s)] = 1
    mask_b                            = cv2.GaussianBlur(mask,(s,s),0)
    mask_hb                           = cv2.GaussianBlur(mask,(21,21),0) 
    arr_sc                            = arr_s.copy()
    arr_sc[mask_b>=10**-10]           = np.NaN
    while np.nanmax(arr_sc)-np.nanmin(arr_sc) >= 100:
        s = s+2
        mask_b                        = cv2.GaussianBlur(mask,(s,s),0)
        arr_sc[mask_b>=10**-10]       = np.NaN
    mask_hb                           = cv2.GaussianBlur(mask,(s+6,s+6),0) 
    sort                              = np.unique(arr_sc[~np.isnan(arr_sc)])
    sortofmedian                      = sort[int(len(sort)/2)]
    std                               = np.std(arr_s[~np.isnan(arr_sc)])
    cap                               = sortofmedian + 1.5*std
    arr_scc                           = arr_sc.copy()
    arr_scc[arr_scc>=cap]             = cap
    arr_sccg                          = arr_scc.copy()
    un                                = np.nanmin(arr_sccg)
    arr_sccg                          = arr_sccg-un
    up                                = np.nanmax(arr_sccg)
    arr_sccg                          = (arr_sccg/up)*255
    arr_sccgb                         = cv2.GaussianBlur(arr_sccg,(3,3),0)
    arr_sccgbf                        = arr_sccgb.copy()
    arr_sccgbf[np.isnan(arr_sccgbf)]  = np.nanmean(arr_sccgbf)
    arr_sccgbf                        = arr_sccgbf.astype(np.uint8)    
    arr_sccgbfa                       = cv2.adaptiveThreshold(arr_sccgbf,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
    arr_sccgbfa[mask_hb>=10**-10]     = 0
    edges                             = cv2.medianBlur(arr_sccgbfa,5)
    exec("edges"+str(i)+" = edges")

#%% IMAGE SCALING [2019-08-08]
def calc_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 1000 * 6371 * c
    return m

def pix_size(array,gt):
    lon1 = gt[0] 
    lat1 = gt[3] 
    lon2 = gt[0] + gt[1]*array.shape[0]
    lat2 = gt[3] + gt[4]*array.shape[0]
    dist = calc_distance(lat1,lon1,lat2,lon2)
    ysize = dist/array.shape[0]  
    lon2 = gt[0] + gt[2]*array.shape[1]
    lat2 = gt[3] + gt[5]*array.shape[1]
    dist = calc_distance(lat1,lon1,lat2,lon2)
    xsize = dist/array.shape[1]
    return xsize, ysize

#x_s, y_s = pix_size(array,gt)
#arr_s    = cv2.resize(array,(int(array.shape[1]*(y_s/0.5)), int(array.shape[0]*(x_s/0.5))),interpolation = cv2.INTER_AREA)


#%% RGB EDGE DETECTION [2019-08-08]
wdir = r"E:"
files = ["0","1"]  

path = []
for x in range(len(files)):
    path.append(wdir+"\\"+files[x]+".tif")
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
#%%
path = path[0]
luma = 709
def ortho_to_edges(path,luma):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = pix_size(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    Rlin                               = (R_s**2.2)/255
    Glin                               = (G_s**2.2)/255
    Blin                               = (B_s**2.2)/255
    if   luma == 709:
        Y                              = 0.2126*Rlin + 0.7152*Glin + 0.0722*Blin
    elif luma == 601:
        Y                              = 0.299*Rlin + 0.587*Glin + 0.114*Blin
    elif luma == 240:
        Y                              = 0.212*Rlin + 0.701*Glin + 0.087*Blin
    L                                  = 116*Y**(1/3)-16
    arr_sg                             = ((L/np.max(L))*255).astype(np.uint8)
    arr_sgb                            = cv2.medianBlur(arr_sg,3)
    mask                               = np.zeros(arr_sg.shape)
    mask[arr_sg==255]                  = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)
    ht                                 = 250
    lt                                 = 0.5*ht
    edges                              = cv2.Canny(arr_sgb,lt,ht)
    edges[mask_b>=10**-10]             = 0 
    fact_x = B.shape[0]/edges.shape[0]
    fact_y = B.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, gt, fact_x, fact_y, x_b, y_b, mask

def dem_to_edges(path):
    s                                 = 5
    file                              = gdal.Open(path)
    band                              = file.GetRasterBand(1)
    gt                                = file.GetGeoTransform()
    arr                               = band.ReadAsArray()
    x_s, y_s                          = pix_size(arr,gt)
    arr_s                             = cv2.resize(arr,(int(arr.shape[1]*(y_s/0.5)), int(arr.shape[0]*(x_s/0.5))),interpolation = cv2.INTER_AREA)
    mask                              = np.zeros(arr_s.shape)
    mask[arr_s<=0.8*np.nanmin(arr_s)] = 1
    mask_b                            = cv2.GaussianBlur(mask,(s,s),0)
    mask_hb                           = cv2.GaussianBlur(mask,(21,21),0) 
    arr_sc                            = arr_s.copy()
    arr_sc[mask_b>=10**-10]           = np.NaN
    while np.nanmax(arr_sc)-np.nanmin(arr_sc) >= 100:
        s = s+2
        mask_b                        = cv2.GaussianBlur(mask,(s,s),0)
        arr_sc[mask_b>=10**-10]       = np.NaN
    mask_hb                           = cv2.GaussianBlur(mask,(s+6,s+6),0) 
    sort                              = np.unique(arr_sc[~np.isnan(arr_sc)])
    sortofmedian                      = sort[int(len(sort)/2)]
    std                               = np.std(arr_s[~np.isnan(arr_sc)])
    cap                               = sortofmedian + 1.5*std
    arr_scc                           = arr_sc.copy()
    arr_scc[arr_scc>=cap]             = cap
    arr_sccg                          = arr_scc.copy()
    un                                = np.nanmin(arr_sccg)
    arr_sccg                          = arr_sccg-un
    up                                = np.nanmax(arr_sccg)
    arr_sccg                          = (arr_sccg/up)*255
    arr_sccgb                         = cv2.GaussianBlur(arr_sccg,(3,3),0)
    arr_sccgbf                        = arr_sccgb.copy()
    arr_sccgbf[np.isnan(arr_sccgbf)]  = np.nanmean(arr_sccgbf)
    arr_sccgbf                        = arr_sccgbf.astype(np.uint8)    
    arr_sccgbfa                       = cv2.adaptiveThreshold(arr_sccgbf,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
    arr_sccgbfa[mask_hb>=10**-10]     = 0
    edges                             = cv2.medianBlur(arr_sccgbfa,5) 
    fact_x = arr.shape[0]/edges.shape[0]
    fact_y = arr.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, gt, fact_x, fact_y, x_b, y_b, mask

#%%
edges, gt, fact_x, fact_y, x_b, y_b, mask = ortho_to_edges(path[0],709)

#%%
plt.imshow(edges)