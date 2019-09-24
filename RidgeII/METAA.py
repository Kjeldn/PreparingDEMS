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

def InboxxFiles(num):
    plt.close("all")
    metapath = []
    for i in range(num):
        path = []
        root = Tk()
        root.withdraw()
        #
        root.filename =  filedialog.askopenfilename(initialdir = r"D:\VanBovenDrive\VanBoven MT\Archive" ,title = "Select Base Orthomosaic",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
        #
        path.append(root.filename)
        root.filename =  filedialog.askopenfilename(multiple=True, initialdir = r"C:\Users\VanBoven\Documents\100 Ortho Inbox" ,title = "Select Orthomosaics for Geo-Registration",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
        for file in root.filename:
            path.append(file)
        metapath.append(path)
    plist = []
    plt.ioff()
    return metapath,plist

def SelectFiles():
    plt.close("all")
    root = Tk()
    root.withdraw()
    #
    root.filename =  filedialog.askopenfilename(initialdir = "D:\VanBovenDrive\VanBoven MT\Archive" ,title = "Select Base Orthomosaic",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
    #
    base = root.filename
    path0 = base[base.find("Archive"):]
    path1 = path0[path0.find("/")+1:]
    path2 = path1[path1.find("/")+1:]
    path3 = path2[path2.find("/")+1:]
    folder = base[:base.find(path3)]
    path = []
    path.append(base)
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            if ".tif" in name:
                if name not in base:
                    if "_DEM" not in name:
                        if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
                            path.append(os.path.join(root,name).replace("\\","/"))             
    plist = []
    plt.ioff()
    return path,plist

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
    
def OrtOpening(plist,path):
    pbar1 = tqdm(total=1,position=0,desc="OrtOpening")
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    B                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    R                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    ps1 = 0.5
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
    fact_x_ps1                         = B.shape[0]/B_s.shape[0]
    fact_y_ps1                         = B.shape[1]/B_s.shape[1]
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
    pbar1.update(1)
    pbar1.close()
    return plist,img_s, img_b, mask_b, gt, fact_x_ps1, fact_y_ps1

def DemOpening(plist,path,Img0C):
    pbar1 = tqdm(total=1,position=0,desc="DemOpening")
    if "-GR" in path:
        temp = path.strip("-GR.tif")+"_DEM-GR.tif"
    else:
        temp = path.strip(".tif")+"_DEM.tif"
    psF=0.05
    file                               = gdal.Open(temp)
    gt                                 = file.GetGeoTransform()
    dem_o                              = file.GetRasterBand(1).ReadAsArray()
    x_s, y_s                           = calc_pixsize(dem_o,gt)
    mask                               = np.zeros(dem_o.shape)
    if np.sum(dem_o==0) > np.sum(dem_o==np.min(dem_o)):
        mask[dem_o == 0]               = 1
    else:
        mask[dem_o == np.min(dem_o)]   = 1
    dem                                = cv2.resize(dem_o,(int(dem_o.shape[1]*(y_s/psF)), int(dem_o.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA)
    mask                               = cv2.resize(mask,(int(mask.shape[1]*(y_s/psF)), int(mask.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA) 
    fx                                 = dem_o.shape[0]/dem.shape[0]
    fy                                 = dem_o.shape[1]/dem.shape[1] 
    dem_f = cv2.GaussianBlur(dem,(11,11),0)
    smooth = cv2.GaussianBlur(dem_f,(15,15),0)
    ridges = (dem_f-smooth)
    #kernel = np.ones((n,n),np.float32)/(n**2)
    #smooth = cv2.filter2D(dem_f,-1,kernel)
    mask_b = cv2.GaussianBlur(mask,(51,51),0)  
    ridges[mask>10**-10]=0  
    temp1 = np.zeros(ridges.shape)
    temp2 = np.zeros(ridges.shape)
    temp1[ridges<-0.01]=1
    temp2[ridges>-0.11]=1
    ridges = (temp1*temp2).astype(np.uint8) 
    p = plt.figure()
    plt.title('Ridges 0.05m')
    plt.imshow(ridges,cmap='Greys')
    pbar1.update(1)
    pbar1.close()
    plt.close()
    plist.append(p)
    return plist,gt,fx,fy,mask_b,ridges

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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def rangemaker(num,thMeaningfulLength):
    span = int(thMeaningfulLength/5)
    range_array = np.zeros(2*span+1)
    for i in range(len(range_array)):
        range_array[i]=int(num-span+i)
    return range_array

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def CapFigures(i,path,plist):
    dpiset = 1000
    filename = path[i].strip('.tif') + ('_LOG.pdf')
    if os.path.exists(filename.replace("\\","/")):
        os.remove(filename)   
    pp = PdfPages(filename)
    for fig in plist:
        fig.savefig(pp, format='pdf',dpi=dpiset)
    plist = np.array(plist)
    plist = plist[0:2]
    plist = list(plist)
    pp.close()
    return plist
    
    