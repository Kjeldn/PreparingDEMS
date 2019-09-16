import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import gdal
import os
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.interpolate import interp1d
from tempfile import mkstemp
from shutil import move

def SinglMatch(psC,w,md,EdgesC1C,gt1,fx1C,fy1C,EdgesC0C,gt0,fx0C,fy0C,MaskB0C):
    pbar = tqdm(total=1,position=0,desc="RECC      ")
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
    target = EdgesC0C[b3:b4,b1:b2]
    sum_target = np.sum(target)   
    xd = int(round((b4-b3)/2))
    yd = int(round((b2-b1)/2)) 
    buffer = 2*max(xd,yd)
    edges1Ca = np.zeros((EdgesC1C.shape[0]+buffer*2,EdgesC1C.shape[1]+2*buffer))
    edges1Ca[buffer:-buffer,buffer:-buffer] = EdgesC1C
    x_i_0=b3+xd    
    y_i_0=b1+yd
    lat_0 = gt0[3] + gt0[5]*x_i_0*fx0C  
    lon_0 = gt0[0] + gt0[1]*y_i_0*fy0C
    x_i_0_og = int(round((lat_0-gt1[3])/(gt1[5]*fx1C)))
    y_i_0_og = int(round((lon_0-gt1[0])/(gt1[1]*fy1C)))     
    search_wide = np.zeros((2*(max_dist+yd),2*(max_dist+xd)))
    search_wide = edges1Ca[buffer+x_i_0_og-max_dist-xd:buffer+x_i_0_og+max_dist+xd,buffer+y_i_0_og-max_dist-yd:buffer+y_i_0_og+max_dist+yd]
    RECC_total = np.zeros(edges1Ca.shape)
    RECC_over  = np.zeros(edges1Ca.shape)
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
    RECC_total[x_i_0_og-max_dist:x_i_0_og+max_dist,y_i_0_og-max_dist:y_i_0_og+max_dist] = RECC_area
    max_one  = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-1)[-1]
    max_n    = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-4-1)[-4-1]
    y_i      = np.where(RECC_total >= max_one)[1][0]  
    x_i      = np.where(RECC_total >= max_one)[0][0]
    y_n      = np.where(RECC_total >= max_n)[1][0:-1]
    x_n      = np.where(RECC_total >= max_n)[0][0:-1]
    CV1      = sum(np.sqrt(np.square(x_i-x_n)+np.square(y_i-y_n)))/4
    x_offset = (x_i-x_i_0_og)*psC
    y_offset = (y_i-y_i_0_og)*psC
    o_x = x_i
    o_y = y_i
    t_x = x_i_0
    t_y = y_i_0
    pbar.update(1)
    pbar.close()
    print("Status    : ("+str(x_offset)+"m,"+str(y_offset)+"m), CV: "+str(CV1))  
    return x_offset,y_offset,o_x, o_y, t_x, t_y,CV1
    
def PatchMatch(ps2, w, md, edges1F, gt, fx_F, fy_F, edges0F, gt_0, fx_F0, fy_F0, contour_F0,x_offset,y_offset,CV1):
    w = int(w/ps2)
    buffer = 2*w
    edges1Fa = np.zeros((edges1F.shape[0]+buffer*2,edges1F.shape[1]+2*buffer))
    edges1Fa[buffer:-buffer,buffer:-buffer] = edges1F
    if CV1>4:
        md = md/2
    elif CV1>1.5:
        md = md/3
    else:
        md = md/4
    max_dist = int((md)/(ps2))
    contours,hierarchy = cv2.findContours((1-contour_F0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    polygon = Polygon(np.array(biggest_contour[:,0]))
    polygon = polygon.buffer(-w)
    v=w
    while polygon.is_empty or polygon.geom_type == 'MultiPolygon' or polygon.area/ps2<100:
        v -= int(2/ps2)
        polygon = Polygon(np.array(biggest_contour[:,0]))
        polygon = polygon.buffer(-v)
    if v != w:
        print("WARNING   : Polygon-buffer: "+str(v)+" < w...")
    x,y = polygon.exterior.xy   
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]
    fx, fy = interp1d( distance, x ), interp1d( distance, y )
    alpha = np.linspace(0, 1, 200)
    x_regular, y_regular = fx(alpha), fy(alpha)
    grid = []
    for i in range(len(x_regular)):
        grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))  
    cv         = np.zeros(len(grid)) 
    dist_lon   = np.zeros(len(grid))
    dist_lat   = np.zeros(len(grid))
    origin_x   = np.zeros(len(grid))
    origin_y   = np.zeros(len(grid))
    target_lon = np.zeros(len(grid))
    target_lat = np.zeros(len(grid))
    o_x        = np.zeros(len(grid))
    o_y        = np.zeros(len(grid))
    t_x        = np.zeros(len(grid))
    t_y        = np.zeros(len(grid))
    dx         = np.zeros(len(grid))
    dy         = np.zeros(len(grid))
    RECC_total = np.zeros(edges1Fa.shape)
    RECC_over  = np.zeros(edges1Fa.shape)
    RECC_over.fill(np.NaN)
    circle1 = np.zeros((2*w,2*w))
    for x in range(circle1.shape[0]):
        for y in range(circle1.shape[1]):
            if (x-w)**2 + (y-w)**2 < w**2:
                circle1[x,y]=1
    circle2 = np.zeros((2*max_dist,2*max_dist))
    for x in range(circle2.shape[0]):
        for y in range(circle2.shape[1]):
            if (x-max_dist)**2 + (y-max_dist)**2 < max_dist**2:
                circle2[x,y]=1
    circle2[circle2==0]=np.NaN
    for i in tqdm(range(len(grid)),position=0,miniters=int(len(grid)/10),desc="RECC      "):
        x_i_0 = grid[i][1]
        y_i_0 = grid[i][0]
        target = edges0F[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
        if target.shape != (2*w,2*w):
            continue
        target = target*circle1
        sum_target = np.sum(target)   
        lat_0 = gt_0[3] + gt_0[5]*x_i_0*fx_F0  
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fy_F0
        x_i_0_og = int(round((lat_0-gt[3])/(gt[5]*fx_F) + (x_offset/ps2)))
        y_i_0_og = int(round((lon_0-gt[0])/(gt[1]*fy_F) + (y_offset/ps2)))
        search_wide = np.zeros((2*(max_dist+w),2*(max_dist+w)))
        search_wide = edges1Fa[buffer+x_i_0_og-max_dist-w:buffer+x_i_0_og+max_dist+w,buffer+y_i_0_og-max_dist-w:buffer+y_i_0_og+max_dist+w]
        if search_wide.shape != (2*(max_dist+w),2*(max_dist+w)):
            continue
        sum_patch = cv2.filter2D(search_wide,-1,circle1)
        numerator = cv2.filter2D(search_wide,-1,target)
        RECC_wide = numerator / (sum_patch+sum_target)
        RECC_area = RECC_wide[w:-w,w:-w]*circle2
        RECC_total.fill(np.NaN)
        RECC_total[x_i_0_og-max_dist:x_i_0_og+max_dist,y_i_0_og-max_dist:y_i_0_og+max_dist] = RECC_area
        if np.nansum(RECC_over[x_i_0_og-max_dist:x_i_0_og+max_dist,y_i_0_og-max_dist:y_i_0_og+max_dist])==0:
            RECC_over[x_i_0_og-max_dist:x_i_0_og+max_dist,y_i_0_og-max_dist:y_i_0_og+max_dist] = RECC_area
        max_one  = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-1)[-1]
        max_n    = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-4-1)[-4-1]
        y_i      = np.where(RECC_total >= max_one)[1][0]  
        x_i      = np.where(RECC_total >= max_one)[0][0]
        y_n      = np.where(RECC_total >= max_n)[1][0:-1]
        x_n      = np.where(RECC_total >= max_n)[0][0:-1]
        cv[i] = sum(np.sqrt(np.square(x_i-x_n)+np.square(y_i-y_n)))/4
        lon = gt[0] + gt[1]*y_i*fy_F + gt[2]*x_i*fx_F
        lat = gt[3] + gt[4]*y_i*fy_F + gt[5]*x_i*fx_F
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fy_F0 + gt_0[2]*x_i_0*fx_F0
        lat_0 = gt_0[3] + gt_0[4]*y_i_0*fy_F0 + gt_0[5]*x_i_0*fx_F0
        dist_lon[i] = lon_0-lon
        dist_lat[i] = lat_0-lat 
        o_x[i] = x_i
        o_y[i] = y_i
        t_x[i] = x_i_0
        t_y[i] = y_i_0
        dx[i] = (x_i-x_i_0_og)*ps2
        dy[i] = (y_i-y_i_0_og)*ps2
        # For referencing:
        origin_x[i] = x_i*fx_F
        origin_y[i] = y_i*fy_F
        target_lon[i] = lon_0
        target_lat[i] = lat_0
    return origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv, dx, dy

def RemOutlier(origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv, dx, dy, gt, gto):
    size0 = len(o_x)
    indices = np.where(cv>0)[0]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    o_x        = o_x[indices]
    o_y        = o_y[indices]
    t_x        = t_x[indices]
    t_y        = t_y[indices]
    cv         = cv[indices]
    size1=len(o_x)
    if len(o_x[cv<1.5]) >= 50:
        ind = np.where(cv<1.5)[0]
    elif len(o_x[cv<4]) >= 50:
        ind = np.where(cv<4)[0]
    else:
        ind = np.where(cv<np.median(cv))[0]
        print("WARNING   : Not enough points with low CV score.")
    sub_d_x = dx[ind]
    sub_d_y = dy[ind]
    a,b,c = np.histogram2d(sub_d_x,sub_d_y,bins=len(sub_d_x))
    d,e = np.where(a==np.max(a))
    if len(d) > 1:
        i = [0,0]
        binnnn = len(sub_d_x)/10
        while len(i) > 1:
            binnnn -= 1 
            f,g,h = np.histogram2d(sub_d_x,sub_d_y,bins=binnnn)
            i,j = np.where(f==np.max(f))
        diff = (b[d]-g[i])**2 + (c[e]-h[j])**2
        ind = np.where(diff == np.min(diff))[0]
        d = d[ind]
        e = e[ind]    
    x_offset = (b[d]+b[d+1])/2
    y_offset = (c[e]+c[e+1])/2
    delta_x = dx - x_offset
    delta_y = dy - y_offset
    distance = delta_x**2 + delta_y**2
    radius = 1
    indices = np.where(distance <= radius)[0]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    o_x        = o_x[indices]
    o_y        = o_y[indices]
    t_x        = t_x[indices]
    t_y        = t_y[indices]
    cv         = cv[indices]
    size2=len(o_x)  
    print("GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]") 
    origin_x = ((gt[3]+gt[5]*origin_x) - gto[3])/gto[5]
    origin_y = ((gt[0]+gt[1]*origin_y) - gto[0])/gto[1]
    gcplist = []
    for k in range(len(origin_x)): 
        gcplist.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    return origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv, gcplist
      
def Georegistr(i,files,gcplist):
    pbar3 = tqdm(total=1,position=0,desc="Georeg    ")
    temp = files[i][::-1]
    temp2 = temp[:temp.find("/")]
    src = temp2[::-1]
    dest = files[i].strip(".tif")+"_georegR.vrt"  
    if os.path.isfile(dest.replace("\\","/")):
        os.remove(dest)
    temp = gdal.Translate('',files[i],format='VRT',outputSRS= 'EPSG:4326',GCPs=gcplist)
    gdal.Warp(dest,temp,tps=True,resampleAlg='bilinear')
    pattern = "    <SourceDataset relativeToVRT=\"0\"></SourceDataset>"
    subst   = "    <SourceDataset relativeToVRT=\"1\">"+src+"</SourceDataset>"
    fh, abs_path = mkstemp()
    with os.fdopen(fh,'w') as new_file:
        with open(dest) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    os.remove(dest)
    move(abs_path, dest)
    pbar3.update(1)
    pbar3.close()
    