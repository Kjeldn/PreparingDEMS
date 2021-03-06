import METAA

import os
import cv2
import copy
import warnings
import numpy as np
import numpy.matlib
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry.polygon import Polygon
from multiprocessing import cpu_count as cpu
from multiprocessing import set_start_method

warnings.simplefilter(action = "ignore", category = RuntimeWarning)


"""
Takes the edges found in the orthomosaics on a coarse scale and produces one
single match of the whole images in order to find a rough offset that aids the
upcoming fine matching procedure.
---
plist    | list   | List for plots
Edges1C  | 2D arr | Binary map found by CannyLin for orthomosaic up for georegistration
gt1C     | tuple  | Geotransform corresponding to Edges1C
Edges0C  | 2D arr | Binary map found by CannyLin for base orthomosaic
gt0C     | tuple  | Geotransform corresponding to Edges0C
MaskB0C  | 2D arr | Binary map defining edges of base orthomosaic
x_off    | int    | X-offset found in meters
y_off    | int    | Y-offset found in meters
CV1      | float  | Concentration value for the given match
"""
def OneMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C):
    psC = 0.5
    md = 5
    pbar = tqdm(total=1,position=0,desc="RECC(c)   ")
    max_dist = int((md)/psC)
    contours,hierarchy = cv2.findContours((1-MaskB0C).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    polygon = Polygon(np.array(biggest_contour[:,0]))
    x,y=polygon.exterior.xy
    b1 = int(min(x))
    b2 = int(max(x))
    b3 = int(min(y))
    b4 = int(max(y))
    target = Edges0C[b3:b4,b1:b2]
    sum_target = np.sum(target)
    xd = int(round((b4-b3)/2))
    yd = int(round((b2-b1)/2))
    buffer = 2*max(xd,yd)
    edges1Ca = np.zeros((Edges1C.shape[0]+buffer*2,Edges1C.shape[1]+2*buffer))
    edges1Ca[buffer:-buffer,buffer:-buffer] = Edges1C
    x0=b3+xd
    y0=b1+yd
    lat_0 = gt0C[3] + gt0C[5]*x0
    lon_0 = gt0C[0] + gt0C[1]*y0
    xog = int(round((lat_0-gt1C[3])/(gt1C[5])))
    yog = int(round((lon_0-gt1C[0])/(gt1C[1])))
    search_wide = np.zeros((2*(max_dist+yd),2*(max_dist+xd)))
    search_wide = edges1Ca[buffer+xog-max_dist-xd:buffer+xog+max_dist+xd,buffer+yog-max_dist-yd:buffer+yog+max_dist+yd]
    RECC_total = np.zeros(Edges1C.shape)
    RECC_over  = np.zeros(Edges1C.shape)
    RECC_over.fill(np.NaN)
    circle2 = np.zeros((2*max_dist,2*max_dist))
    for x in range(circle2.shape[0]):
        for y in range(circle2.shape[1]):
            if (x-max_dist)**2 + (y-max_dist)**2 < max_dist**2:
                circle2[x,y]=1
    circle2[circle2==0]=np.NaN
    sum_patch = cv2.filter2D(search_wide,-1,np.ones(target.shape))
    numerator = cv2.filter2D(search_wide,-1,target)
    RECC_wide = numerator / (sum_patch+sum_target)
    RECC_area = RECC_wide[xd:-xd,yd:-yd]*circle2
    RECC_total.fill(np.NaN)
    if RECC_total[xog-max_dist:xog+max_dist,yog-max_dist:yog+max_dist].shape != (2*(max_dist),2*(max_dist)):
        pbar.update(1)
        pbar.close()
        x_off = 0
        y_off = 0
        x1=0
        y1=0
        CV1=100
    else:
        RECC_total[xog-max_dist:xog+max_dist,yog-max_dist:yog+max_dist] = RECC_area
        max_one  = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-1)[-1]
        max_n    = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-4-1)[-4-1]
        y1       = np.where(RECC_total >= max_one)[1][0]
        x1       = np.where(RECC_total >= max_one)[0][0]
        y_n      = np.where(RECC_total >= max_n)[1][0:-1]
        x_n      = np.where(RECC_total >= max_n)[0][0:-1]
        CV1      = sum(np.sqrt(np.square(x1-x_n)+np.square(y1-y_n)))/4
        x_off = (x1-xog)*psC
        y_off = (y1-yog)*psC
        pbar.update(1)
        pbar.close()
        p = plt.figure()
        plt.figtext(.8, .8, "[R] Origin \n[G] Other Grid \n[B] Offset")
        plt.subplot(1,2,1)
        plt.title("Offset: ("+str(x_off)+","+str(y_off)+") m")
        plt.imshow(Edges0C,cmap='Greys')
        plt.scatter(y0,x0,c='r',s=3)
        plt.subplot(1,2,2)
        plt.title("CV: "+str(round(CV1,2)))
        plt.imshow(Edges1C,cmap='Greys')
        plt.scatter(y0,x0,c='r',s=1)
        plt.plot([y0,yog],[x0,xog],c='g',lw=0.1,alpha=0.5)
        plt.scatter(yog,xog,c='g',s=1)
        plt.plot([yog,y1],[xog,x1],c='b',lw=0.1,alpha=0.5)
        plt.scatter(y1,x1,c='b',s=1)
        plt.close()
        plist.append(p)
    return plist,x_off,y_off,CV1


"""
Setup certain variables before matching GCPs in DEM data. Define the grid,
buffer Edges1F, define w and md, and create clipping circles.
---
plist    | list   | List for plots
Edges0F  | 2D arr | Binary map from DEM up for georegistration
Edges1F  | 2D arr | Binary map from DEM corresponding to base
MaskB0F  | 2D arr | Binary map defining edges of orthomosaic
x_off    | int    | X-offset found by OneMatch function using Edges0C and Edges1C
y_off    | int    | Y-offset found by OneMatch function using Edges0C and Edges1C
CV1      | float  | Concentration value for the match produced by OneMatch
Edges1Fa | 2D arr | Buffered version of Edges1F to prevent out of bounds
grid     | list   | List of 200 or 300 x,y tuples that form a grid in Edges0F
md       | int    | Maximum distance, the maximum feasable error in pixels
c1       | 2D arr | Binary circle for clipping a patch
c2       | 2D arr | Binary circle for clipping of a search map
"""
def IniMatch(plist,Edges0F,Edges1F,MaskB0F,MaskB1F,x_off,y_off,CV1,gt0F,gt1F):
    # Nullify impact of OpenOrth + CannyLin + OneMatch:
    x_off = 0
    y_off = 0
    CV1 = 1.5
    
    ps0F = 0.05
    w = int(25/ps0F)
    buff = w
    Edges1Fa = np.zeros((Edges1F.shape[0]+buff*2,Edges1F.shape[1]+2*buff))
    Edges1Fa[buff:-buff,buff:-buff] = Edges1F
    Edges1Fa=(Edges1Fa).astype(np.uint8)
    if CV1>=4:
        md = 5
        x_off=0;y_off=0
    elif CV1<=1.5:
        md = 2
    else:
        md = 2 + 3*((CV1-1.5)/2.5)
    md=6
    md = int(6/ps0F)
    
    contours,hierarchy = cv2.findContours((1-MaskB0F).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    poly_base = Polygon(np.array(biggest_contour[:,0]))
    contours,hierarchy = cv2.findContours((1-MaskB1F).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    poly_file = Polygon(np.array(biggest_contour[:,0]))
    if poly_file.area < 0.7*poly_base.area:
        sf=1
        print("Georegistration of small orthomosaic on larger, very cool, preventive move to nrtu folder afterwards...")
        inward = 0
        polygon = poly_file.buffer(-w)
        while polygon.area < 0.6*poly_file.area:
            inward +=20 
            polygon = poly_file.buffer(-w+inward)
        while polygon.type == 'MultiPolygon':
            polygon = sorted(list(polygon), key=lambda p:p.area, reverse=True)[0]
        x,y = polygon.exterior.xy
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        alpha = np.linspace(0, 1, 200)
        x_regular, y_regular = fx(alpha), fy(alpha)
        grid = []
        for i in range(len(x_regular)):
            lat = gt1F[3] + gt1F[5]*y_regular[i]
            lon = gt1F[0] + gt1F[1]*x_regular[i]
            y_regular[i] = (lat-gt0F[3])/gt0F[5]
            x_regular[i] = (lon-gt0F[0])/gt0F[1]
            grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
    elif poly_base.area < 0.4*poly_file.area:
        sf=1
        print("Georegistration of large orthomosaic on smaller, moving to nrtu folder afterwards...")
        polygon = poly_base.buffer(-w)
        while polygon.type == 'MultiPolygon':
            polygon = sorted(list(polygon), key=lambda p:p.area, reverse=True)[0]
        x,y = polygon.exterior.xy
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        alpha = np.linspace(0, 1, 200)
        x_regular, y_regular = fx(alpha), fy(alpha)
        grid = []
        for i in range(len(x_regular)):
            grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
        if polygon.buffer(-3*w).is_empty == False:
            polygon = polygon.buffer(-2*w)
            while polygon.type == 'MultiPolygon':
                polygon = sorted(list(polygon), key=lambda p:p.area, reverse=True)[0]
            x,y = polygon.exterior.xy
            distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
            distance = distance/distance[-1]
            fx, fy = interp1d( distance, x ), interp1d( distance, y )
            alpha = np.linspace(0, 1, 100)
            x_regular, y_regular = fx(alpha), fy(alpha)
            for i in range(len(x_regular)):
                grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
    else:    
        sf=0
        polygon = poly_base.buffer(-w)
        while polygon.type == 'MultiPolygon':
            polygon = sorted(list(polygon), key=lambda p:p.area, reverse=True)[0]
        x,y = polygon.exterior.xy
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        alpha = np.linspace(0, 1, 200)
        x_regular, y_regular = fx(alpha), fy(alpha)
        grid = []
        for i in range(len(x_regular)):
            grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
        if polygon.buffer(-3*w).is_empty == False:
            polygon = polygon.buffer(-2*w)
            while polygon.type == 'MultiPolygon':
                polygon = sorted(list(polygon), key=lambda p:p.area, reverse=True)[0]
            x,y = polygon.exterior.xy
            distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
            distance = distance/distance[-1]
            fx, fy = interp1d( distance, x ), interp1d( distance, y )
            alpha = np.linspace(0, 1, 100)
            x_regular, y_regular = fx(alpha), fy(alpha)
            for i in range(len(x_regular)):
                grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))                
    c1 = np.zeros((2*w,2*w))
    for x in range(c1.shape[0]):
        for y in range(c1.shape[1]):
            if (x-w)**2 + (y-w)**2 < w**2:
                c1[x,y]=1
    c1 = (c1).astype(np.uint8)
    c2 = np.zeros((2*md,2*md))
    for x in range(c2.shape[0]):
        for y in range(c2.shape[1]):
            if (x-md)**2 + (y-md)**2 < md**2:
                c2[x,y]=1
    c2 = (c2).astype(np.uint8)
    return plist,Edges1Fa,x_off,y_off,grid,md,c1,c2,sf


"""
Function that controls the actual matching of each grid point to a patch in
the DEM up for georegistration. Takes output from IniMatch as input, divides
the grid in a number of batches, which are distributed over a number of logical
processors. The actual matching is carried out by BatchMatch.
---
plist    | list   | List for plots
Edges0F  | 2D arr | Binary map from DEM up for georegistration
Edges1F  | 2D arr | Binary map from DEM corresponding to base
Edges1Fa | 2D arr | Buffered version of Edges1F to prevent out of bounds
x_off    | int    | X-offset found by OneMatch function using Edges0C and Edges1C
y_off    | int    | Y-offset found by OneMatch function using Edges0C and Edges1C
CV1      | float  | Concentration value for the match produced by OneMatch
grid     | list   | List of 200 or 300 x,y tuples that form a grid in Edges0F
md       | int    | Maximum distance, the maximum feasable error in pixels
c1       | 2D arr | Binary circle for clipping a patch
c2       | 2D arr | Binary circle for clipping of a search map
gt0F     | tuple  | Geotransform corresponding to Edges0F
gt1F     | tuple  | Geotransform corresponding to Edges1F
x0       | 1D arr | Array containing x-pixel in Edges0F
y0       | 1D arr | Array containing y-pixel in Edges0F
x1       | 1D arr | Array containing x-pixel in Edges1F corresponding to x0
y1       | 1D arr | Array containing y-pixel in Edges1F corresponding to y0
CV       | 1D arr | Array with concentration values corresponding to each (x0,y0) - (x1,y1) match
dx       | 1D arr | Array containing x-offset in meters for each match
dy       | 1D arr | Array containing y-offset in meters for each match
"""
def MulMatch(plist,Edges0F,Edges1F,Edges1Fa,x_off,y_off,CV1,grid,md,circle1,circle2,gt0F,gt1F):
    set_start_method('spawn', force=True)
    ps0F = 0.05
    w = int(25/ps0F)
    from RECCM import BatchMatch as bm
    func = partial(bm,w,md,Edges0F,Edges1F,Edges1Fa,circle1,circle2,gt0F,gt1F,x_off,y_off)
    num_workers = cpu()
    pool = Pool(num_workers)
    bpw = 2
    num_batches = int(num_workers*bpw)
    while len(grid)/num_batches > 10:
        bpw += 1
        num_batches = int(num_workers*bpw)
    N=int(np.ceil(len(grid)/num_batches))
    pbar = tqdm(total=num_batches+1,position=0,desc="RECC(f)   ")
    x0=[];y0=[];xog=[];yog=[];xof=[];yof=[];x1=[];y1=[];CV=[];dx=[];dy=[];
    results = pool.imap_unordered(func,(grid[int(i*N):int((i+1)*N)] for i in range(num_batches)),chunksize=N)
    for re in results:
        x0.extend(re[0]);y0.extend(re[1]);xog.extend(re[2]);yog.extend(re[3]);xof.extend(re[4]);yof.extend(re[5]);x1.extend(re[6]);y1.extend(re[7]);CV.extend(re[8]);dx.extend(re[9]);dy.extend(re[10]);
        pbar.update(1)
    pool.close()
    pool.join()
    x0=np.array(x0);y0=np.array(y0);xog=np.array(xog);yog=np.array(yog);xof=np.array(xof);yof=np.array(yof);x1=np.array(x1);y1=np.array(y1);CV=np.array(CV);dx=np.array(dx);dy=np.array(dy)
    p = plt.figure()
    plt.subplot(1,2,1)
    plt.title("Patch")
    plt.imshow(Edges0F,cmap='Greys')
    plt.scatter(y0,x0,c='r',s=1)
    plt.subplot(1,2,2)
    plt.title("Match")
    plt.imshow(Edges1F,cmap='Greys')
    plt.figtext(.8, 0.8, "[R] Origin \n[G] Other Grid \n[B] Offset \n[Y] Patch Match")
    plt.scatter(y0,x0,c='r',s=1)
    for i in range(len(y1)):
        plt.plot([y0[i],yog[i]],[x0[i],xog[i]],c='g',lw=0.1,alpha=0.5)
    plt.scatter(yog,xog,c='g',s=1)
    for i in range(len(y1)):
        plt.plot([yog[i],yof[i]],[xog[i],xof[i]],c='b',lw=0.1,alpha=0.5)
    plt.scatter(yof,xof,c='b',s=1)
    for i in range(len(y1)):
        if x1[i] != 0 and y1[i] != 0:
            plt.plot([yof[i],y1[i]],[xof[i],x1[i]],c='y',lw=0.1,alpha=0.5)
    ind = np.where(x1!=0)
    plt.scatter(y1[ind],x1[ind],c='y',s=1)
    plt.close()
    plist.append(p)
    pbar.update(1)
    pbar.close()
    return plist,x0,y0,x1,y1,CV,dx,dy


"""
Function that carries out the Relative Edge Cross Correlation (RECC) matching
for a part of the grid. Note therefore that the grid passed to this function is
in fact a slice of the original grid. Each (x,y) is transformed to the
corresponding (x,y) in the other DEM, then the offset is applied, and finally
a match is found by considering a region around that last point, and convolving
the target with a search region according to the RECC formula.
---
w        | int    | Radius of the circular patch
md       | int    | Maximum distance, the maximum feasable error in pixels
Edges0F  | 2D arr | Binary map from DEM up for georegistration
Edges1F  | 2D arr | Binary map from DEM corresponding to base
Edges1Fa | 2D arr | Buffered version of Edges1F to prevent out of bounds
c1       | 2D arr | Binary circle for clipping a patch
c2       | 2D arr | Binary circle for clipping of a search map
gt0F     | tuple  | Geotransform corresponding to Edges0F
gt1F     | tuple  | Geotransform corresponding to Edges1F
x_off    | int    | X-offset found by OneMatch function using Edges0C and Edges1C
y_off    | int    | Y-offset found by OneMatch function using Edges0C and Edges1C
grid     | list   | List of 200 or 300 x,y tuples that form a grid in Edges0F
x0       | 1D arr | Array containing x-pixel in Edges0F
y0       | 1D arr | Array containing y-pixel in Edges0F
xog      | 1D arr | Array containing x-pixel in Edges1F on top of x0 via lat-lon transform
yog      | 1D arr | Array containing y-pixel in Edges1F on top of y0 via lat-lon transform
xof      | 1D arr | Array containing x-pixel in Edges1F shifted from xog based on x_off
yof      | 1D arr | Array containing y-pixel in Edges1F shifted from yog based on y_off
x1       | 1D arr | Array containing x-pixel in Edges1F corresponding to x0 by actual matching
y1       | 1D arr | Array containing y-pixel in Edges1F corresponding to y0 by actual matching
CV       | 1D arr | Array with concentration values corresponding to each (x0,y0) - (x1,y1) match
dx       | 1D arr | Array containing x-offset in meters for each match
dy       | 1D arr | Array containing y-offset in meters for each match
"""
def BatchMatch(w,md,Edges0F,Edges1F,Edges1Fa,c1,c2,gt0F,gt1F,x_off,y_off,grid):
    buff       = w
    CV         = np.zeros(len(grid))
    x0         = np.zeros(len(grid)).astype(int)
    y0         = np.zeros(len(grid)).astype(int)
    xog        = np.zeros(len(grid)).astype(int)
    yog        = np.zeros(len(grid)).astype(int)
    xof        = np.zeros(len(grid)).astype(int)
    yof        = np.zeros(len(grid)).astype(int)
    x1         = np.zeros(len(grid)).astype(int)
    y1         = np.zeros(len(grid)).astype(int)
    lat        = np.zeros(len(grid))
    lon        = np.zeros(len(grid))
    for i in range(len(grid)):
        x0[i] = grid[i][1]
        y0[i] = grid[i][0]
        lat[i] = gt0F[3] + gt0F[5]*x0[i]
        lon[i] = gt0F[0] + gt0F[1]*y0[i]
        xog[i] = int(round((lat[i]-gt1F[3])/(gt1F[5])))
        yog[i] = int(round((lon[i]-gt1F[0])/(gt1F[1])))
        xof[i] = int(round(xog[i] + x_off/0.05))
        yof[i] = int(round(yog[i] + y_off/0.05))
        target = copy.deepcopy(Edges0F[x0[i]-w:x0[i]+w,y0[i]-w:y0[i]+w])
        if target.shape != (2*w,2*w):
            continue
        target[c1==0] = 0
        sum_target = np.sum(target)
        search_wide = Edges1Fa[buff+xof[i]-md-w:buff+xof[i]+md+w,buff+yof[i]-md-w:buff+yof[i]+md+w]
        if search_wide.shape != (2*(md+w),2*(md+w)) or np.sum(search_wide) == 0:
            continue
        sum_patch = cv2.filter2D(search_wide.astype(float),-1,c1.astype(float))
        numerator = cv2.filter2D(search_wide.astype(float),-1,target.astype(float))
        RECC_wide = numerator / (sum_patch+sum_target)
        RECC_area = RECC_wide[w:-w,w:-w]
        RECC_area[c2==0]=np.NaN
        if RECC_area.shape != (2*md,2*md):
            continue
        try:
            max_one  = np.partition(RECC_area[~np.isnan(RECC_area)].flatten(),-1)[-1]
            max_n    = np.partition(RECC_area[~np.isnan(RECC_area)].flatten(),-4-1)[-4-1]
        except:
            continue
        x,y = np.where(RECC_area >= max_one)
        y1[i]    = y[0]
        x1[i]    = x[0]
        x,y = np.where(RECC_area >= max_n)
        y_n = y[0:-1]
        x_n = x[0:-1]
        CV[i] = sum(np.sqrt(np.square(x1[i]-x_n)+np.square(y1[i]-y_n)))/4
        x1[i] = x1[i] + xof[i]-md
        y1[i] = y1[i] + yof[i]-md
    dx = (x1-xof)*0.05
    dy = (y1-yof)*0.05
    return x0,y0,xog,yog,xof,yof,x1,y1,CV,dx,dy


"""
Outlier removal procedure. First it is checked whether gridpoints have gone
out of bounds by lat-lon tranformation to the other DEM. After this, a second
order fit is created in an iterative manner. On the final iteration only points
that have a (dx,dy) offset close enough to the fitted function are kept.
---
plist    | list   | List for plots
Edges0F  | 2D arr | Binary map from DEM up for georegistration
Edges1F  | 2D arr | Binary map from DEM corresponding to base
x0       | 1D arr | Array containing x-pixel in Edges0F
y0       | 1D arr | Array containing y-pixel in Edges0F
x1       | 1D arr | Array containing x-pixel in Edges1F corresponding to x0
y1       | 1D arr | Array containing y-pixel in Edges1F corresponding to y0
CV       | 1D arr | Array with concentration values corresponding to each (x0,y0) - (x1,y1) match
dx       | 1D arr | Array containing x-offset in meters for each match
dy       | 1D arr | Array containing y-offset in meters for each match
GCPstat  | tuple  | Contains the status of matches or GCP's after outlier removal
"""
def RemovOut(plist,Edges0F,Edges1F,x0,y0,x1,y1,CV,dx,dy):
    size0 = len(x0)
    indices = np.where(CV>0)[0]
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CV        = CV[indices]
    dx        = dx[indices]
    dy        = dy[indices]
    size1=len(x0)
    clist = list(np.random.choice(range(256), size=len(x0)))
    p=plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(Edges0F,cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow(Edges1F,cmap='Greys')
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title(str(size1)+" in-domain")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close()
    plist.append(p)
    clist = np.array(clist)
    indices = np.where(CV>-1)[0]
    dist_range = [1.5,1,0.5,0.25,0.1]
    for dist in dist_range:
        fdx = METAA.hifit(x0[indices],y0[indices],CV[indices],dx[indices])
        fdy = METAA.hifit(x0[indices],y0[indices],CV[indices],dy[indices])
        supposed_dx = fdx[0]*x0+fdx[1]*y0+fdx[2]*(x0*y0)+fdx[3]*(x0**2)+fdx[4]*(y0**2)+fdx[5]
        supposed_dy = fdy[0]*x0+fdy[1]*y0+fdy[2]*(x0*y0)+fdy[3]*(x0**2)+fdy[4]*(y0**2)+fdy[5]
        delta_x = dx - supposed_dx
        delta_y = dy - supposed_dy
        distance = np.sqrt(np.square(delta_x) + np.square(delta_y))
        radius = dist
        indices = np.where(distance.T <= radius)[0]
    inv_indices = []
    for i in range(len(dx)):
        if i not in indices:
            inv_indices.append(i)
    p = plt.figure()
    ax = p.add_subplot(111, projection='3d')
    ax.scatter(x0,y0,supposed_dx,c='b',marker='o',alpha=0.2)
    ax.scatter(x0[indices],y0[indices],dx[indices],c='g',marker='o')
    ax.scatter(x0[inv_indices],y0[inv_indices],dx[inv_indices],c='r',marker='o')
    ax.set_zlim(min(dx)-0.05, max(dx)+0.05)
    plt.close()
    plist.append(p)
    p = plt.figure()
    ax = p.add_subplot(111, projection='3d')
    ax.scatter(x0,y0,supposed_dy,c='b',marker='o',alpha=0.2)
    ax.scatter(x0[indices],y0[indices],dy[indices],c='g',marker='o')
    ax.scatter(x0[inv_indices],y0[inv_indices],dy[inv_indices],c='r',marker='o')
    ax.set_zlim(min(dy)-0.05, max(dy)+0.05)
    plt.close()
    plist.append(p)
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CV        = CV[indices]
    clist     = clist[indices]
    size2=len(x0)
    GCPstat = "GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]"
    print(GCPstat)
    YN = 0
    if size2/size1 > 0.2:
        YN = 1
    GCPstat = (YN,GCPstat)
    clist = list(clist)
    p = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(Edges0F,cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow(Edges1F,cmap='Greys')
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title("("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close()
    plist.append(p)
    return plist,x0,y0,x1,y1,CV,dx,dy,GCPstat


"""
Convert (x0,y0) - (x1,y1) to (lat,lon) - (lat,lon) tranformation and place it
in a .points file, such that it can be used to georegister both the orthomosaic
and the DEM.
---
path     | str    | Path to the orthomosaic up for georegistration
x0       | 1D arr | Array containing x-pixel in Edges0F
y0       | 1D arr | Array containing y-pixel in Edges0F
x1       | 1D arr | Array containing x-pixel in Edges1F corresponding to x0
y1       | 1D arr | Array containing y-pixel in Edges1F corresponding to y0
gt0F     | tuple  | Geotransform corresponding to Edges0F
gt1F     | tuple  | Geotransform corresponding to Edges1F
f1       | int    | Flag for creation of .points file
"""
def MakePnts(path,x0,y0,x1,y1,gt0F,gt1F):
    pbar3 = tqdm(total=1,position=0,desc="MakePoints")
    f1=0
    target_lat = gt0F[3]+gt0F[5]*x0
    target_lon = gt0F[0]+gt0F[1]*y0
    origin_lat = gt1F[3]+gt1F[5]*x1
    origin_lon = gt1F[0]+gt1F[1]*y1
    dest = path.strip(".tif")+".points"
    if os.path.isfile(dest.replace("\\","/")):
        os.remove(dest)
    f = open(dest,"w+")
    f.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual")
    for i in range(len(target_lon)):
        f.write("\n"+str(target_lon[i])+","+str(target_lat[i])+","+str(origin_lon[i])+","+str(origin_lat[i])+",1,0,0,0")    
        f1=len(x0)
    f.close()
    pbar3.update(1)
    pbar3.close()
    return f1


"""
###############################################################################
Archived functions below.
Mostly replaced by improved functions.
###############################################################################
"""


#def PatchMatch(plist,Edges1F, gt1F, fx1F, fy1F, Edges0F, gt0F, fx0F, fy0F, MaskB0F,x_off,y_off,CV1):
#    ps0F = 0.05
#    w = int(25/ps0F)
#    buffer = 2*w
#    edges1Fa = np.zeros((Edges1F.shape[0]+buffer*2,Edges1F.shape[1]+2*buffer))
#    edges1Fa[buffer:-buffer,buffer:-buffer] = Edges1F
#    if CV1>4:
#        md = 10
#    elif CV1<1.5:
#        md = 5
#    else:
#        md = 5 + 5*((CV1-1.5)/2.5)
#    max_dist = int((md)/(ps0F))
#    contours,hierarchy = cv2.findContours((1-MaskB0F).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
#    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
#    polygon = Polygon(np.array(biggest_contour[:,0]))
#    ref_area = polygon.area
#    polygon = polygon.buffer(-w)
#    v=w
#    while polygon.is_empty or polygon.geom_type == 'MultiPolygon' or polygon.area<0.4*ref_area:
#        v -= int(2/ps0F)
#        polygon = Polygon(np.array(biggest_contour[:,0]))
#        polygon = polygon.buffer(-v)
#    if v != w:
#        print("WARNING   : Polygon-buffer: "+str(v*ps0F)+" < 25...")
#    x,y = polygon.exterior.xy
#    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
#    distance = distance/distance[-1]
#    fx, fy = interp1d( distance, x ), interp1d( distance, y )
#    alpha = np.linspace(0, 1, 200)
#    x_regular, y_regular = fx(alpha), fy(alpha)
#    grid = []
#    for i in range(len(x_regular)):
#        grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
#    if polygon.buffer(-2*w).is_empty == False:
#        polygon = polygon.buffer(-2*w)
#        x,y = polygon.exterior.xy
#        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
#        distance = distance/distance[-1]
#        fx, fy = interp1d( distance, x ), interp1d( distance, y )
#        alpha = np.linspace(0, 1, 100)
#        x_regular, y_regular = fx(alpha), fy(alpha)
#        for i in range(len(x_regular)):
#            grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))
#    CVa        = np.zeros(len(grid))
#    x0         = np.zeros(len(grid)).astype(int)
#    y0         = np.zeros(len(grid)).astype(int)
#    xog        = np.zeros(len(grid)).astype(int)
#    yog        = np.zeros(len(grid)).astype(int)
#    xof        = np.zeros(len(grid)).astype(int)
#    yof        = np.zeros(len(grid)).astype(int)
#    x1         = np.zeros(len(grid)).astype(int)
#    y1         = np.zeros(len(grid)).astype(int)
#    origin_x   = np.zeros(len(grid))
#    origin_y   = np.zeros(len(grid))
#    target_lon = np.zeros(len(grid))
#    target_lat = np.zeros(len(grid))
#    RECC_total = np.zeros(Edges1F.shape)
#    circle1 = np.zeros((2*w,2*w))
#    for x in range(circle1.shape[0]):
#        for y in range(circle1.shape[1]):
#            if (x-w)**2 + (y-w)**2 < w**2:
#                circle1[x,y]=1
#    circle2 = np.zeros((2*max_dist,2*max_dist))
#    for x in range(circle2.shape[0]):
#        for y in range(circle2.shape[1]):
#            if (x-max_dist)**2 + (y-max_dist)**2 < max_dist**2:
#                circle2[x,y]=1
#    circle2[circle2==0]=np.NaN
#    for i in tqdm(range(len(grid)),position=0,miniters=int(len(grid)/10),desc="RECC(f)   "):
#        x0[i] = grid[i][1]
#        y0[i] = grid[i][0]
#        target_lon[i] = gt0F[0] + gt0F[1]*y0[i]*fy0F
#        target_lat[i] = gt0F[3] + gt0F[5]*x0[i]*fx0F
#        xog[i] = int(round((target_lat[i]-gt1F[3])/(gt1F[5]*fx1F)))
#        yog[i] = int(round((target_lon[i]-gt1F[0])/(gt1F[1]*fy1F)))
#        xof[i] = int(round(xog[i] + x_off/ps0F))
#        yof[i] = int(round(yog[i] + y_off/ps0F))
#        target = Edges0F[x0[i]-w:x0[i]+w,y0[i]-w:y0[i]+w]
#        if target.shape != (2*w,2*w):
#            continue
#        target = target*circle1
#        sum_target = np.sum(target)
#        search_wide = np.zeros((2*(max_dist+w),2*(max_dist+w)))
#        search_wide = edges1Fa[buffer+xof[i]-max_dist-w:buffer+xof[i]+max_dist+w,buffer+yof[i]-max_dist-w:buffer+yof[i]+max_dist+w]
#        if search_wide.shape != (2*(max_dist+w),2*(max_dist+w)):
#            continue
#        sum_patch = cv2.filter2D(search_wide,-1,circle1)
#        numerator = cv2.filter2D(search_wide,-1,target)
#        RECC_wide = numerator / (sum_patch+sum_target)
#        RECC_area = RECC_wide[w:-w,w:-w]*circle2
#        RECC_total.fill(np.NaN)
#        if RECC_total[xof[i]-max_dist:xof[i]+max_dist,yof[i]-max_dist:yof[i]+max_dist].shape != (2*(max_dist),2*(max_dist)):
#            continue
#        RECC_total[xof[i]-max_dist:xof[i]+max_dist,yof[i]-max_dist:yof[i]+max_dist] = RECC_area
#        max_one  = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-1)[-1]
#        max_n    = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-4-1)[-4-1]
#        y1[i]    = np.where(RECC_total >= max_one)[1][0]
#        x1[i]    = np.where(RECC_total >= max_one)[0][0]
#        y_n      = np.where(RECC_total >= max_n)[1][0:-1]
#        x_n      = np.where(RECC_total >= max_n)[0][0:-1]
#        CVa[i] = sum(np.sqrt(np.square(x1[i]-x_n)+np.square(y1[i]-y_n)))/4
#        origin_x[i] = x1[i]*fx1F
#        origin_y[i] = y1[i]*fy1F
#    dx = (x1-xof)*ps0F
#    dy = (y1-yof)*ps0F
#    p = plt.figure()
#    plt.subplot(1,2,1)
#    plt.title("Patch")
#    plt.imshow(Edges0F,cmap='Greys')
#    plt.scatter(y0,x0,c='r',s=1)
#    plt.subplot(1,2,2)
#    plt.title("Match")
#    plt.imshow(Edges1F,cmap='Greys')
#    plt.figtext(.8, 0.8, "[R] Origin \n[G] Other Grid \n[B] Offset \n[Y] Patch Match")
#    plt.scatter(y0,x0,c='r',s=1)
#    for i in range(len(y1)):
#        plt.plot([y0[i],yog[i]],[x0[i],xog[i]],c='g',lw=0.1,alpha=0.5)
#    plt.scatter(yog,xog,c='g',s=1)
#    for i in range(len(y1)):
#        plt.plot([yog[i],yof[i]],[xog[i],xof[i]],c='b',lw=0.1,alpha=0.5)
#    plt.scatter(yof,xof,c='b',s=1)
#    for i in range(len(y1)):
#        if x1[i] != 0 and y1[i] != 0:
#            plt.plot([yof[i],y1[i]],[xof[i],x1[i]],c='y',lw=0.1,alpha=0.5)
#    ind = np.where(x1!=0)
#    plt.scatter(y1[ind],x1[ind],c='y',s=1)
#    plt.close()
#    plist.append(p)
#    plt.figure(257)
#    plt.subplot(1,2,1)
#    plt.imshow(Edges0F,cmap='Greys')
#    plt.subplot(1,2,2)
#    plt.imshow(Edges1F,cmap='Greys')
#    plt.figure(258)
#    plt.subplot(1,2,1)
#    plt.imshow(Edges0F,cmap='Greys')
#    plt.subplot(1,2,2)
#    plt.imshow(Edges1F,cmap='Greys')
#    return plist,origin_x,origin_y,target_lon,target_lat,x0,y0,xog,yog,xof,yof,x1,y1,CVa,dx,dy
#
#def RemOutlier(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,files,iiii):
#    size0 = len(x0)
#    indices = np.where(CVa>0)[0]
#    origin_x   = origin_x[indices]
#    origin_y   = origin_y[indices]
#    target_lon = target_lon[indices]
#    target_lat = target_lat[indices]
#    x0        = x0[indices]
#    y0        = y0[indices]
#    x1        = x1[indices]
#    y1        = y1[indices]
#    CVa       = CVa[indices]
#    dx        = dx[indices]
#    dy        = dy[indices]
#    size1=len(x0)
#    clist = list(np.random.choice(range(256), size=len(x0)))
#    p=plt.figure(257)
#    plt.subplot(1,2,1)
#    plt.title("GCP Status:")
#    plt.scatter(y0,x0,s=1,c=clist)
#    plt.subplot(1,2,2)
#    plt.title(str(size1)+" in-domain")
#    plt.scatter(y1,x1,s=1,c=clist)
#    plt.close(257)
#    plist.append(p)
#    clist = np.array(clist)
#    if len(x0[CVa<1.5]) >= 50:
#        ind = np.where(CVa<1.5)[0]
#    elif len(x0[CVa<4]) >= 50:
#        ind = np.where(CVa<4)[0]
#    else:
#        ind = np.where(CVa<np.median(CVa))[0]
#        print("WARNING   : Not enough points with low CV score.")
#    sub_d_x = dx[ind]
#    sub_d_y = dy[ind]
#    a,b,c = np.histogram2d(sub_d_x,sub_d_y,bins=len(sub_d_x))
#    d,e = np.where(a==np.max(a))
#    if len(d) > 1:
#        i = [0,0]
#        binnnn = len(sub_d_x)/10
#        while len(i) > 1:
#            binnnn -= 1
#            f,g,h = np.histogram2d(sub_d_x,sub_d_y,bins=binnnn)
#            i,j = np.where(f==np.max(f))
#        diff = (b[d]-g[i])**2 + (c[e]-h[j])**2
#        ind = np.where(diff == np.min(diff))[0]
#        d = d[ind]
#        e = e[ind]
#    x_offset = (b[d]+b[d+1])/2
#    y_offset = (c[e]+c[e+1])/2
#    delta_x = dx - x_offset
#    delta_y = dy - y_offset
#    distance = delta_x**2 + delta_y**2
#    radius = 1
#    indices = np.where(distance <= radius)[0]
#    origin_x   = origin_x[indices]
#    origin_y   = origin_y[indices]
#    target_lon = target_lon[indices]
#    target_lat = target_lat[indices]
#    x0        = x0[indices]
#    y0        = y0[indices]
#    x1        = x1[indices]
#    y1        = y1[indices]
#    CVa       = CVa[indices]
#    clist     = clist[indices]
#    size2=len(x0)
#    print("GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")
#    gcplist_DEM = []
#    for k in range(len(origin_x)):
#        gcplist_DEM.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
#    gto = gdal.Open(files[iiii]).GetGeoTransform()
#    origin_x = ((gt1F[3]+gt1F[5]*origin_x) - gto[3])/gto[5]
#    origin_y = ((gt1F[0]+gt1F[1]*origin_y) - gto[0])/gto[1]
#    gcplist = []
#    for k in range(len(origin_x)):
#        gcplist.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
#    clist = list(clist)
#    p = plt.figure(258)
#    plt.subplot(1,2,1)
#    plt.title("GCP Status:")
#    plt.scatter(y0,x0,s=1,c=clist)
#    plt.subplot(1,2,2)
#    plt.title("("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")
#    plt.scatter(y1,x1,s=1,c=clist)
#    plt.close(258)
#    plist.append(p)
#    return plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist,gcplist_DEM
#
#def Georegistr(i,files,gcplist,gcplist_DEM):
#    if len(gcplist) != 0:
#        pbar3 = tqdm(total=2,position=0,desc="Georeg    ")
#        temp = files[i][::-1]
#        temp2 = temp[:temp.find("/")]
#        src = temp2[::-1]
#        dest = files[i].strip(".tif")+"_GR.vrt"
#        if os.path.isfile(dest.replace("\\","/")):
#            os.remove(dest)
#        temp = gdal.Translate('',files[i],format='VRT',outputSRS= 'EPSG:4326',GCPs=gcplist)
#        gdal.Warp(dest,temp,tps=True,resampleAlg='bilinear')
#        pattern = "    <SourceDataset relativeToVRT=\"0\"></SourceDataset>"
#        subst   = "    <SourceDataset relativeToVRT=\"1\">"+src+"</SourceDataset>"
#        fh, abs_path = mkstemp()
#        with os.fdopen(fh,'w') as new_file:
#            with open(dest) as old_file:
#                for line in old_file:
#                    new_file.write(line.replace(pattern, subst))
#        os.remove(dest)
#        move(abs_path, dest)
#        pbar3.update(1)
#
#        file = files[i].strip(".tif")+"_DEM.tif"
#        temp = file[::-1]
#        temp2 = temp[:temp.find("/")]
#        src = temp2[::-1]
#        dest = file.strip(".tif")+"_GR.vrt"
#        if os.path.isfile(dest.replace("\\","/")):
#            os.remove(dest)
#        temp = gdal.Translate('',file,format='VRT',outputSRS= 'EPSG:4326',GCPs=gcplist_DEM)
#        gdal.Warp(dest,temp,tps=True,resampleAlg='bilinear')
#        pattern = "    <SourceDataset relativeToVRT=\"0\"></SourceDataset>"
#        subst   = "    <SourceDataset relativeToVRT=\"1\">"+src+"</SourceDataset>"
#        fh, abs_path = mkstemp()
#        with os.fdopen(fh,'w') as new_file:
#            with open(dest) as old_file:
#                for line in old_file:
#                    new_file.write(line.replace(pattern, subst))
#        os.remove(dest)
#        move(abs_path, dest)
#        pbar3.update(1)
#        pbar3.close()
#
#    medx0 = (np.max(x0)-np.min(x0))/2 + np.min(x0)
#    medy0 = (np.max(y0)-np.min(y0))/2 + np.min(y0)
#    x0 = x0 - medx0
#    y0 = y0 - medy0
#
#    A = []
#    bdx = []
#    bdy = []
#    for i in range(len(x0[ind])):
#        A.append([x0[ind][i], y0[ind][i], x0[ind][i]*y0[ind][i], x0[ind][i]**2, y0[ind][i]**2, 1])
#        bdx.append(dx[ind][i])
#        bdy.append(dy[ind][i])
#    bdx = np.matrix(bdx).T
#    bdy = np.matrix(bdy).T
#    A = np.matrix(A)
#    W = np.diag(1/CVa[ind])
#    W = (W/np.sum(W))*len(CVa)
#
#    ErrorFunc_dx=lambda fit: np.ravel(W*(bdx - np.dot(A,fit)))
#    ErrorFunc_dy=lambda fit: np.ravel(W*(bdy - np.dot(A,fit)))
#
#    fdx = [0,0,0,0,0,np.median(dx[ind])]
#    fdx,flagx = optimize.leastsq(ErrorFunc_dx,fdx)
#    fdy = [0,0,0,0,0,np.median(dy[ind])]
#    fdy,flagy = optimize.leastsq(ErrorFunc_dy,fdy)
#
#    ErrorFunc_dx=lambda fit: np.sum((W*(bdx - np.dot(A,fit)))**2)
#    ErrorFunc_dy=lambda fit: np.sum((W*(bdy - np.dot(A,fit)))**2)
#
#
#    fdx = [1,1,1,1,1,np.median(dx[ind])]
#    fdx = optimize.minimize(ErrorFunc_dx,fdx,method='Nelder-Mead',tol=0.1,options={'maxiter':100,'maxfun':500,'adaptive':True})
#    fdy = [0,0,0,0,0,np.median(dy[ind])]
#    fdy = optimize.minimize(ErrorFunc_dy,fdy,options={'maxiter':5000}).x
#
#    supposed_dx = fdx[0]*x0+fdx[1]*y0+fdx[2]*(x0*y0)+fdx[3]*(x0**2)+fdx[4]*(y0**2)+fdx[5]
#    supposed_dy = fdy[0]*x0+fdy[1]*y0+fdy[2]*(x0*y0)+fdy[3]*(x0**2)+fdy[4]*(y0**2)+fdy[5]
#
#    x0 = x0 + medx0
#    y0 = y0 + medy0
