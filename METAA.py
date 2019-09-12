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
import copy
import sys
import os
from tkinter import filedialog
from tkinter import *

def Initialize():
    psC = 0.5   #[m]  (0.5)  <First pixelsize>
    psF = 0.05  #[m]  (0.05) <Second pixelsize>
    w   = 25    #[m]  (25)   <Radius template>
    md  = 12    #[m]  (12)   <Max displacement>
    return psC,psF,w,md
    
    
def LogStarter(l,i,files):
    orig = "no log active..."
    if l == 1:
        folder = files[i].strip(".tif") + "_georeg"
        logname = folder+"/Z_log.txt"
        if os.path.isfile(logname.replace("\\","/")):
            os.remove(logname)
        if not os.path.exists(folder.replace("\\","/")):
            os.makedirs(folder)
        orig = sys.stdout
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush() # If you want the output to be visible immediately
            def flush(self) :
                for f in self.files:
                    f.flush()
        f = open(logname, 'w')
        sys.stdout = Tee(sys.stdout, f)
    return(orig)

def LogStopper(orig):
    sys.stdout = orig 

def OrthOpenin(ps1,ps2,path):
    pbar1 = tqdm(total=1,position=0,desc="Opening   ")
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    B                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    R                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    fact_x_ps1                         = B.shape[0]/B_s.shape[0]
    fact_y_ps1                         = B.shape[1]/B_s.shape[1]
    #x_b_ps1                            = B_s.shape[0]
    #y_b_ps1                            = B_s.shape[1]
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
    fact_x_ps2                         = B.shape[0]/B_s.shape[0]
    fact_y_ps2                         = B.shape[1]/B_s.shape[1]
    #x_b_ps2                            = B_s.shape[0]
    #y_b_ps2                            = B_s.shape[1]
    pbar1.update(1)
    pbar1.close()
    return gt, img_s, img_b, mask_b, fact_x_ps1, fact_y_ps1, img_s2, fact_x_ps2, fact_y_ps2

def OrthSwitch(ps2,ps1,img_s2,edgeChainsE):
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
    return img_b, mask_n, mask_o

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

def intersect(a1, a2, b1, b2):
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

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

def next3(xSeed,ySeed,rows,cols,residualmap,boe,s,edgeChain):
    if s >= 3:
        if boe == 0:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#,-1, 1,-1, 1, 0]         
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]#, 0, 0, 1, 1, 1]
        elif boe == 1:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]#, 0, 0,-1,-1,-1]
    elif s < 3 and s > 1/3:
        if boe == 0:
            # 6
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2]#, 1,-1]#, 1, 0, 1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0]#,-1, 1]#, 0, 1, 1]
        elif boe == 1:
            # 2
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2]#,-1, 1]#,-1, 0,-1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0]#, 1,-1]#, 0,-1,-1]
    elif s >= -1/3 and s <= 1/3:
        if boe == 0:
            # 7
            X_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]#, 0, 0, 1, 1, 1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#,-1, 1, 1,-1, 0]
        elif boe == 1:
            # 3
            X_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]#, 0, 0,-1,-1,-1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#, 1,-1,-1, 1, 0]
    elif s < -1/3 and s > -3:
        if boe == 0:
            # 8
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2]#, 1,-1]#, 1, 0, 1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0]#, 1,-1]#, 0,-1,-1]
        elif boe == 1:
            # 4
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2]#,-1, 1]#,-1, 0,-1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0]#,-1, 1]#, 0, 1, 1]
    elif s <= -3:
        if boe == 0:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2]#, 0, 0,-1,-1,-1]
        elif boe == 1:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2]#,-1, 1,-1, 1, 0]  
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2]#, 0, 0, 1, 1, 1]
    a=-1
    b=-1
    for i in range(len(X_OFFSET)):
        x = xSeed + X_OFFSET[i]
        if (x >= 0) and (x < rows):
            y = ySeed + Y_OFFSET[i]
            if (y >= 0) and (y < cols):
                if residualmap[x,y] == 1 and (x,y) not in edgeChain:
                    a = x
                    b = y
                    break
    return a, b

def next4(xSeed,ySeed,rows,cols,residualmap,boe,s,edgeChain):
    INNER_RING_X = np.array([ 0, 1, 1, 1, 0,-1,-1,-1])
    INNER_RING_Y = np.array([ 1, 1, 0,-1,-1,-1, 0, 1])
    OUTER_RING_X = np.array([ 0, 1, 2, 2, 2, 2, 2, 1, 0,-1,-2,-2,-2,-2,-2,-1])
    OUTER_RING_Y = np.array([ 2, 2, 2, 1, 0,-1,-2,-2,-2,-2,-2,-1, 0, 1, 2, 2])
    if boe == 1:
        if s >= 2:
            inner = 0
        elif 0.5 <= s < 2:
            inner = 1
        elif -0.5 <= s < 0.5:
            inner = 2
        elif -2 <= s < -0.5:
            inner = 3
        elif s < -2:
            inner = 4
        if s >= 4:
            outer = 0
        elif 1.33 <= s < 4:
            outer = 1
        elif 0.75 <= s < 1.33:
            outer = 2
        elif 0.25 <= s < 0.75:
            outer = 3
        elif -0.25 <= s < 0.25:
            outer = 4
        elif -0.75 <= s < -0.25:
            outer = 5
        elif -1.33 <= s < -0.75:
            outer = 6
        elif -4 <= s < -1.33:
            outer = 7
        elif s < -4:
            outer = 8
    if boe == 0:
        if s >= 2:
            inner = 4
        elif 0.5 <= s < 2:
            inner = 5
        elif -0.5 <= s < 0.5:
            inner = 6
        elif -2 <= s < -0.5:
            inner = 7
        elif s < -2:
            inner = 0
        if s >= 4:
            outer = 8
        elif 1.33 <= s < 4:
            outer = 9
        elif 0.75 <= s < 1.33:
            outer = 10
        elif 0.25 <= s < 0.75:
            outer = 11
        elif -0.25 <= s < 0.25:
            outer = 12
        elif -0.75 <= s < -0.25:
            outer = 13
        elif -1.33 <= s < -0.75:
            outer = 14
        elif -4 <= s < -1.33:
            outer = 15
        elif s < -4:
            outer = 0
    randomint = randint(-1,1)
    while randomint == 0:
        randomint = randint(-1,1)
    inner_indices = [inner, inner+randomint, inner-randomint]
    outer_indices = [outer, outer-randomint, outer+randomint]
    for i in range(0,3):
        if inner_indices[i] == -1:
            inner_indices[i] = 7
        if inner_indices[i] == 8:
            inner_indices[i] = 0
    for i in range(0,3):
        if outer_indices[i] == -1:
            outer_indices[i] = 15
        if outer_indices[i] == 16:
            outer_indices[i] = 0
    X_OFFSET = list(INNER_RING_X[inner_indices]) + list(OUTER_RING_X[outer_indices])
    Y_OFFSET = list(INNER_RING_Y[inner_indices]) + list(OUTER_RING_Y[outer_indices])
    a=-1
    b=-1
    for i in range(len(X_OFFSET)):
        x = xSeed + X_OFFSET[i]
        if (x >= 0) and (x < rows):
            y = ySeed + Y_OFFSET[i]
            if (y >= 0) and (y < cols):
                if residualmap[x,y] == 1 and (x,y) not in edgeChain:
                    a = x
                    b = y
                    break
    return a, b

def rangemaker(num,thMeaningfulLength):
    span = int(thMeaningfulLength/5)
    range_array = np.zeros(2*span+1)
    for i in range(len(range_array)):
        range_array[i]=int(num-span+i)
    return range_array
        
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def overlay(img):
    temp = copy.deepcopy(img)
    temp[temp==0]=np.NaN
    return temp

def CreateFigs(logi,i,files,psC,psF,w,Img0C,Img1C,ImgB0C,ImgB1C,Img0F,Img1F,ImgB0F,ImgB1F,EdgesA0C,EdgesB0C,EdgesC0C,EdgesA1C,EdgesB1C,EdgesC1C,EdgesA0F,EdgesB0F,EdgesC0F,EdgesA1F,EdgesB1F,EdgesC1F,X0C,Y0C,X1C,Y1C,X0Fa,Y0Fa,X1Fa,Y1Fa,X0Fb,Y0Fb,X1Fb,Y1Fb,CVb):
    pbar3 = tqdm(total=1,position=0,desc="LogFigures")
    folder = files[i].strip(".tif") + "_georeg"
    if not os.path.exists(folder.replace("\\","/")):
        os.makedirs(folder)
    plt.close("all")
    # [CANNY] (Coarse) Edge Check
    ind = min(np.where(CVb==min(CVb))[0])
    x = X0Fb[ind]/(psC/psF)
    y = Y0Fb[ind]/(psC/psF)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='winter')
    plt.imshow(overlay(EdgesA0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\A_Edges0C_a.png",dpi = 500)
    plt.figure()
    ind = min(np.where(CVb==max(CVb))[0])
    x = X0Fb[ind]/(psC/psF)
    y = Y0Fb[ind]/(psC/psF)
    plt.subplot(2,2,1)
    plt.imshow(ImgB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='winter')
    plt.imshow(overlay(EdgesA0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC0C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\A_Edges0C_b.png",dpi = 500)
    # [CANNY] (Coarse) Edge Check
    ind = min(np.where(CVb==min(CVb))[0])
    x = X0Fb[ind]/(psC/psF)
    y = Y0Fb[ind]/(psC/psF)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='winter')
    plt.imshow(overlay(EdgesA1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')    
    if logi == 1:
        plt.savefig(folder+"\\B_Edges1C_a.png",dpi = 500)
    ind = min(np.where(CVb==max(CVb))[0])
    x = X0Fb[ind]/(psC/psF)
    y = Y0Fb[ind]/(psC/psF)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='winter')
    plt.imshow(overlay(EdgesA1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC1C[int(x-w/psC):int(x+w/psC),int(y-w/psC):int(y+w/psC)]),cmap='gray') 
    if logi == 1:
        plt.savefig(folder+"\\B_Edges1C_b.png",dpi = 500)
    # [RECC] (Coarse) GCP
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(Img0C)  
    plt.imshow(overlay(EdgesC0C),cmap='gist_rainbow')
    plt.scatter(Y0C,X0C,c='b',s=2)
    plt.subplot(1,2,2)
    plt.imshow(Img1C)  
    plt.imshow(overlay(EdgesC1C),cmap='gist_rainbow')
    plt.plot([Y0C,Y1C],[X0C,X1C],c='y',lw=0.5)
    plt.scatter(Y1C,X1C,c='b',s=2)  
    if logi == 1:
        plt.savefig(folder+"\\C_SingleMatch.png",dpi = 500)
    # [CANNY] (Fine) Edge Check
    ind = min(np.where(CVb==min(CVb))[0])
    x = X0Fb[ind]
    y = Y0Fb[ind]
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='winter')
    plt.imshow(overlay(EdgesA0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\D_Edges0F_a",dpi = 500) 
    ind = min(np.where(CVb==max(CVb))[0])
    x = X0Fb[ind]
    y = Y0Fb[ind]
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='winter')
    plt.imshow(overlay(EdgesA0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC0F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\D_Edges0F_b",dpi = 500)
    # [CANNY] (Fine) Edge Check
    ind = min(np.where(CVb==min(CVb))[0])
    x = X0Fb[ind]
    y = Y0Fb[ind]
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='winter')
    plt.imshow(overlay(EdgesA1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\E_Edges1F_a",dpi = 500)
    ind = min(np.where(CVb==max(CVb))[0])
    x = X0Fb[ind]
    y = Y0Fb[ind]
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ImgB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(overlay(EdgesA1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(overlay(EdgesB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='winter')
    plt.imshow(overlay(EdgesA1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(overlay(EdgesB1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gist_rainbow')
    plt.imshow(overlay(EdgesC1F[int(x-w/psF):int(x+w/psF),int(y-w/psF):int(y+w/psF)]),cmap='gray')
    if logi == 1:
        plt.savefig(folder+"\\E_Edges1F_b",dpi = 500)
    # [RECC] (Fine) GCP original
    plt.figure()
    clist = list(np.random.choice(range(256), size=len(X0Fa)))
    plt.subplot(1,2,1)
    plt.title('Orthomosaic 0')
    plt.imshow(Img0F)  
    plt.scatter(Y0Fa,X0Fa,c=clist,s=5)
    plt.subplot(1,2,2)
    plt.title('Orthomosaic 1')
    plt.imshow(Img1F)  
    plt.scatter(Y1Fa,X1Fa,c=clist,s=5)  
    if logi == 1:
        plt.savefig(folder+"\\F_RECCa",dpi = 500)
    # [RECC] (Fine) GCP outlier
    plt.figure()
    clist = list(np.random.choice(range(256), size=len(X0Fb)))
    plt.subplot(1,2,1)
    plt.title('Orthomosaic 0')
    plt.imshow(Img0F)  
    plt.scatter(Y0Fb,X0Fb,c=clist,s=5)
    plt.subplot(1,2,2)
    plt.title('Orthomosaic 1')
    plt.imshow(Img1F)  
    plt.scatter(Y1Fb,X1Fb,c=clist,s=5)
    if logi == 1:
        plt.savefig(folder+"\\G_RECCb",dpi = 500)
    # [RECC] RECC check
    #fig,ax = plt.subplots()
    #plt.imshow(RECC_over,cmap='gray')
    #ax.scatter(Y1Fb,X1Fb,c='r')
    #for i in range(len(X1Fb)):
    #    ax.annotate(str(round(CVb[i],2)),(Y1Fb[i]+(7/0.05),X1Fb[i]-(7/0.05)))
    plt.close("all")
    pbar3.update(1)
    pbar3.close()

def SelectBase(): 
    root = Tk()
    root.withdraw()
    #
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select Base Orthomosaic",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
    #
    temp = root.filename[::-1]
    temp2 = temp[temp.find("/")+1:]
    wdir = temp2[::-1]
    return root.filename,wdir

def SelectFiles(wdir):  
    root = Tk()
    root.withdraw()
    #
    root.filename =  filedialog.askopenfilename(multiple=True,initialdir = wdir,title = "Select Orthomosaics for Georegistration",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
    #
    return root.filename