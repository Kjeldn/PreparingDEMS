wdir = r"E:\ORTHODUMP"
files = ["T0","T1","T2","E0","E1"]
files = ["T0","T1","T2"]

file_type      = 0         # [0: Orthomosaic] [1: DEM]
gamma_correct  = 1         # {0,1}
luma           = 601       # {240,601,709}
extra_blur     = 1         # {0,1}
thresholding   = 0         # {0: Binary Otsu} {1: Median} {2: Mean}
sigma          = 1/3
cv_max         = 40        # Approve matches only if CV score is under {<...>} .
dst_max        = 20        # Approve matches only if distance is under {<...>} [m] .
v              = 30        # Masking square will have a length of {2*<...> pixel_size} [m].
s              = 4         # RECC search range distance is capped at {<...>*w*pixel_size} [m].
it_cancel      = 1         # {0,1} 
it_max         = 5         # Max number of iterations per match
outlier_type   = 1         # [0: Boxplot] [1: 2D Confidence]
conf           = 75        # {50,75,80,90,95}

pixel_size     = 0.1       # One pixel will be <...> m by <...> m.
ppp            = 25        # Use <...> points per photo to georeference.
w              = 200        # Search square will have a length of {2*<...>*pixel_size} [m].

import rasterio
import gdal
import cv2
import numpy.matlib
import matplotlib.pyplot as plt
import os
import numpy as np
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import RECC

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
    
#%%
path,steps = RECC.initialize(wdir,files,ppp)
for i in range(len(path)):
    print(i)
    exec("edges_"+str(i)+", array_"+str(i)+", gt_"+str(i)+", fact_x_"+str(i)+", fact_y_"+str(i)+", x_b_"+str(i)+", y_b_"+str(i)+", mask_"+str(i)+" =  RECC.ortho_to_color(path[i],pixel_size)")
     
#%%
for i in range(1,len(path)):
    o_x        = np.zeros(ppp)
    o_y        = np.zeros(ppp)
    t_x        = np.zeros(ppp)
    t_y        = np.zeros(ppp)
    j=-1
    while j <= ppp-2:
        print(j)
        x_b_0 = array_0.shape[0]
        y_b_0 = array_0.shape[1]
        x_i_0 = randint(w,x_b_0-w)
        y_i_0 = randint(w,y_b_0-w)
        check1 = mask_0[x_i_0,y_i_0]
        if check1 <= 0:
             target = array_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
             exec("res = cv2.matchTemplate(array_"+str(i)+",target,cv2.TM_CCOEFF)")
             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
             center = (max_loc[0] + w, max_loc[1] + w)
             o_x[j] = center[0]
             o_y[j] = center[1]
             t_x[j] = x_i_0
             t_y[j] = y_i_0
             j=j+1
    exec("o_x_"+str(i)+" = o_x")
    exec("o_y_"+str(i)+" = o_y")
    exec("t_x_"+str(i)+" = t_x")
    exec("t_y_"+str(i)+" = t_y")

clist = list(np.random.choice(range(256), size=ppp))
#%%
plt.imshow(array_0)
plt.scatter(t_y_2,t_x_2,c=clist)

#%%
plt.imshow(array_2)
plt.scatter(o_x_2,o_y_2,c=clist)