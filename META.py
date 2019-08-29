import gdal
import cv2
import numpy as np
import numpy.matlib
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm

def initialize(wdir,files):
    path = []
    for x in range(len(files)):
        path.append(wdir+"\\"+files[x]+".tif")
    return path

def correct_ortho(ps1,ps2,path):
    pbar1 = tqdm(total=1,position=0,desc="Opening   ")
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    img_s                              = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
    mask                               = np.zeros(B_s.shape)
    mask[R_s==255]                     = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)  
    img_s[:,:,0]                       = B_s
    img_s[:,:,1]                       = G_s
    img_s[:,:,2]                       = R_s
    img_s_cielab                       = cv2.cvtColor(img_s, cv2.COLOR_BGR2Lab)
    L                                  = img_s_cielab[:,:,0] 
    hist                               = np.histogram(L[mask_b==0],bins=256)[0]
    cdf                                = hist.cumsum()
    cdf_m                              = np.ma.masked_equal(cdf,0)
    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
    L_eq                               = cdf[L] 
    img_s_cielab_eq                    = img_s_cielab.copy()
    img_s_cielab_eq[:,:,0]             = L_eq   
    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
    img_g                              = cv2.cvtColor(img_s_eq, cv2.COLOR_BGR2GRAY)
    fsize                              = int(np.ceil((1.05/ps1))//2*2+1)
    img_b                              = cv2.bilateralFilter(img_g,fsize,125,250)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/ps2)), int(B.shape[0]*(x_s/ps2))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/ps2)), int(B.shape[0]*(x_s/ps2))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/ps2)), int(B.shape[0]*(x_s/ps2))),interpolation = cv2.INTER_AREA)
    img_s2                             = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
    img_s2[:,:,0]                      = B_s
    img_s2[:,:,1]                      = G_s
    img_s2[:,:,2]                      = R_s
    fact_x = B.shape[0]/B_s.shape[0]
    fact_y = B.shape[1]/B_s.shape[1]
    x_b    = B_s.shape[0]
    y_b    = B_s.shape[1]
    pbar1.update(1)
    pbar1.close()
    return img_s, img_g, img_b, mask_b, gt, img_s2, fact_x, fact_y, x_b, y_b

def switch_correct_ortho(ps1,ps2,img_s2,edgeChainsE):
    pbar2 = tqdm(total=1,position=0,desc="Switching ")
    ratio                              = int(ps1/ps2)
    mask_o                             = np.zeros(img_s2[:,:,0].shape)
    mask_o[img_s2[:,:,0]==255]  = 1
    mask_o_b                           = cv2.GaussianBlur(mask_o,(5,5),0) 
    mask_n                             = np.zeros(img_s2[:,:,0].shape)
    for chain in edgeChainsE:
        for point in chain:
            mask_n[(point[0]-1)*ratio:(point[0]+2)*ratio,(point[1]-1)*ratio:(point[1]+2)*ratio]=1
    mask_n[mask_o_b>=10**-10]          = 0
    img_s3                             = np.zeros(img_s2.shape,np.uint8)
    img_s3[:,:,0]                      = img_s2[:,:,0]*mask_n
    img_s3[:,:,1]                      = img_s2[:,:,1]*mask_n
    img_s3[:,:,2]                      = img_s2[:,:,2]*mask_n
    img_s3_cielab                      = cv2.cvtColor(img_s3,cv2.COLOR_BGR2Lab)
    L                                  = img_s3_cielab[:,:,0]
    hist                               = np.histogram(L[mask_n==1],bins=256)[0]
    cdf                                = hist.cumsum()
    cdf_m                              = np.ma.masked_equal(cdf,0)
    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
    L_eq                               = cdf[L] 
    img_s_cielab_eq                    = img_s3_cielab.copy()
    img_s_cielab_eq[:,:,0]             = L_eq   
    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
    img_g                              = cv2.cvtColor(img_s_eq, cv2.COLOR_BGR2GRAY)    
    img_b                              = img_g
    mask_n                             = 1 - mask_n
    pbar2.update(1)
    pbar2.close()
    return img_s_eq, img_g, img_b, mask_n, mask_o

def calc_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 1000 * 6371 * c
    return m

def calc_pixsize(array,gt):
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

def next1(xSeed,ySeed,rows,cols,maskMap,orientationMap):
    X_OFFSET = [0, 1, 0,-1, 1,-1,-1, 1]
    Y_OFFSET = [1, 0,-1, 0, 1, 1,-1,-1]
    direction = orientationMap[xSeed,ySeed]
    direction0 = direction-1
    if direction0 < 0:
        direction0 = 15
    direction1 = direction
    direction2 = direction + 1
    if direction2 == 16:
        direction2 = 0 
    a=-1
    b=-1
    for i in range(0,8):
        x = xSeed + X_OFFSET[i]
        if (x >= 0) and (x < rows):
            y = ySeed + Y_OFFSET[i]
            if (y >= 0) and (y < cols):
                if maskMap[x,y] == 1:
                    directionTemp = orientationMap[x,y]
                    if (directionTemp == direction1) or (directionTemp == direction0) or (directionTemp == direction2):
                        a = x
                        b = y
                        break
    return a, b

def next2(xSeed,ySeed,rows,cols,residualmap,boe,s):
    if s >= 3:
        if boe == 0:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]         
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]
        elif boe == 1:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]
    elif s < 3 and s > 1/3:
        if boe == 0:
            # 6
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0,-1, 1]
        elif boe == 1:
            # 2
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0, 1,-1]
    elif s >= -1/3 and s <= 1/3:
        if boe == 0:
            # 7
            X_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]
        elif boe == 1:
            # 3
            X_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]
    elif s < -1/3 and s > -3:
        if boe == 0:
            # 8
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0,-1, 1]
        elif boe == 1:
            # 4
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0, 1,-1]
    elif s <= -3:
        if boe == 0:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]
        elif boe == 1:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]         
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]     
    a=-1
    b=-1
    for i in range(len(X_OFFSET)):
        x = xSeed + X_OFFSET[i]
        if (x >= 0) and (x < rows):
            y = ySeed + Y_OFFSET[i]
            if (y >= 0) and (y < cols):
                if residualmap[x,y] == 1:
                    a = x
                    b = y
                    break
    return a, b

def next3(xSeed,ySeed,rows,cols,residualmap,boe,s):
    if s >= 3:
        if boe == 0:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]
        elif boe == 1:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]
    elif s < 3 and s > 1/3:
        if boe == 0:
            # 6
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1, 1, 0, 1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0,-1, 1, 0, 1, 1]
        elif boe == 1:
            # 2
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1,-1, 0,-1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0, 1,-1, 0,-1,-1]
    elif s >= -1/3 and s <= 1/3:
        if boe == 0:
            # 7
            X_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1, 1,-1, 0]
        elif boe == 1:
            # 3
            X_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2, 1,-1,-1, 1, 0]
    elif s < -1/3 and s > -3:
        if boe == 0:
            # 8
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1, 1, 0, 1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0, 1,-1, 0,-1,-1]
        elif boe == 1:
            # 4
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1,-1, 0,-1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0,-1, 1, 0, 1, 1]
    elif s <= -3:
        if boe == 0:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]
        elif boe == 1:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]  
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]
    a=-1
    b=-1
    for i in range(len(X_OFFSET)):
        x = xSeed + X_OFFSET[i]
        if (x >= 0) and (x < rows):
            y = ySeed + Y_OFFSET[i]
            if (y >= 0) and (y < cols):
                if residualmap[x,y] == 1:
                    a = x
                    b = y
                    break
    return a, b

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n