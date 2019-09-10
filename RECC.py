import META
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import gdal
import os
from random import randint
import math
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.interpolate import interp1d
             
def init_match(ps1,w,md,edges1C,gt,fx_C,fy_C,xb_C,yb_C,edges0C,gt_0,fx_C0,fy_C0,xb_C0,yb_C0,mask_b_C0):
    w = int(w/ps1)
    buffer = 2*w
    edges1Ca = np.zeros((edges1C.shape[0]+buffer*2,edges1C.shape[1]+2*buffer))
    edges1Ca[buffer:-buffer,buffer:-buffer] = edges1C
    max_dist = int((md)/ps1)
    contours,hierarchy = cv2.findContours((1-mask_b_C0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    polygon = Polygon(np.array(biggest_contour[:,0]))
    polygon = polygon.buffer(-w)
    v=w
    while polygon.is_empty or polygon.geom_type == 'MultiPolygon' or polygon.area/ps1<100:
        v -= int(2/ps1)
        polygon = Polygon(np.array(biggest_contour[:,0]))
        polygon = polygon.buffer(-v)
    if v != w:
        print("WARNING   : Polygon-buffer: "+str(v*ps1)+" < w...")
    x,y = polygon.exterior.xy   
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]
    fx, fy = interp1d( distance, x ), interp1d( distance, y )
    alpha = np.linspace(0, 1, 200)
    x_regular, y_regular = fx(alpha), fy(alpha)
    grid = []
    for i in range(len(x_regular)):
        grid.append((int(round(x_regular[i])),int(round(y_regular[i]))))  
    x_offset   = []
    y_offset   = []
    o_x        = np.zeros(len(grid))
    o_y        = np.zeros(len(grid))
    t_x        = np.zeros(len(grid))
    t_y        = np.zeros(len(grid))
    cv         = np.zeros(len(grid)) 
    RECC_total = np.zeros(edges1Ca.shape)
    RECC_over  = np.zeros(edges1Ca.shape)
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
        target = edges0C[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]*circle1
        if target.shape != (2*w,2*w):
            continue
        sum_target = np.sum(target)   
        lat_0 = gt_0[3] + gt_0[5]*x_i_0*fx_C0  
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fy_C0
        x_i_0_og = int(round((lat_0-gt[3])/(gt[5]*fx_C)))
        y_i_0_og = int(round((lon_0-gt[0])/(gt[1]*fy_C))) 
        search_wide = np.zeros((2*(max_dist+w),2*(max_dist+w)))
        search_wide = edges1Ca[buffer+x_i_0_og-max_dist-w:buffer+x_i_0_og+max_dist+w,buffer+y_i_0_og-max_dist-w:buffer+y_i_0_og+max_dist+w]
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
        x_offset.append(x_i-x_i_0_og)
        y_offset.append(y_i-y_i_0_og)
        o_x[i] = x_i
        o_y[i] = y_i
        t_x[i] = x_i_0
        t_y[i] = y_i_0
    x_offset = np.array(x_offset)
    y_offset = np.array(y_offset)
    a,b,c = np.histogram2d(x_offset[cv<4],y_offset[cv<4],bins=len(x_offset))
    d,e = np.where(a==np.max(a))
    if len(d) > 1:
        i = [0,0]
        binnnn = len(x_offset)/10
        while len(i) > 1:
            binnnn -= 1 
            f,g,h = np.histogram2d(x_offset[cv<4],y_offset[cv<4],bins=binnnn)
            i,j = np.where(f==np.max(f))
        diff = (b[d]-g[i])**2 + (c[e]-h[j])**2
        ind = np.where(diff == np.min(diff))[0]
        d = d[ind]
        e = e[ind]    
    x_offset = (b[d]+b[d+1])/2
    y_offset = (c[e]+c[e+1])/2
    x_offset = x_offset[0]*ps1
    y_offset = y_offset[0]*ps1
    o_x = o_x[cv<4]
    o_y = o_y[cv<4]
    t_x = t_x[cv<4]
    t_y = t_y[cv<4]
    return x_offset,y_offset,o_x, o_y, t_x, t_y

def init_square(ps1,w,md,edges1C,gt,fx_C,fy_C,xb_C,yb_C,edges0C,gt_0,fx_C0,fy_C0,xb_C0,yb_C0,mask_b_C0):
    max_dist = int((md)/ps1)
    contours,hierarchy = cv2.findContours((1-mask_b_C0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    polygon = Polygon(np.array(biggest_contour[:,0]))
    x,y=polygon.exterior.xy
    b1 = int(min(x))
    b2 = int(max(x))
    b3 = int(min(y))
    b4 = int(max(y)) 
    target = edges0C[b3:b4,b1:b2]
    sum_target = np.sum(target)   
    xd = int(round((b4-b3)/2))
    yd = int(round((b2-b1)/2)) 
    buffer = 2*max(xd,yd)
    edges1Ca = np.zeros((edges1C.shape[0]+buffer*2,edges1C.shape[1]+2*buffer))
    edges1Ca[buffer:-buffer,buffer:-buffer] = edges1C
    x_i_0=b3+xd    
    y_i_0=b1+yd
    lat_0 = gt_0[3] + gt_0[5]*x_i_0*fx_C0  
    lon_0 = gt_0[0] + gt_0[1]*y_i_0*fy_C0
    x_i_0_og = int(round((lat_0-gt[3])/(gt[5]*fx_C)))
    y_i_0_og = int(round((lon_0-gt[0])/(gt[1]*fy_C)))     
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
    y_i      = np.where(RECC_total >= max_one)[1][0]  
    x_i      = np.where(RECC_total >= max_one)[0][0]
    x_offset = (x_i-x_i_0_og)*ps1
    y_offset = (y_i-y_i_0_og)*ps1
    o_x = x_i
    o_y = y_i
    t_x = x_i_0
    t_y = y_i_0
    return x_offset,y_offset,o_x, o_y, t_x, t_y
    
def patch_match(ps1, ps2, w, md, edges1F, gt, fx_F, fy_F, xb_F, yb_F, edges0F, gt_0, fx_F0, fy_F0, xb_F0, yb_F0, contour_F0, x_offset, y_offset):
    w = int(w/ps2)
    buffer = 2*w
    edges1Fa = np.zeros((edges1F.shape[0]+buffer*2,edges1F.shape[1]+2*buffer))
    edges1Fa[buffer:-buffer,buffer:-buffer] = edges1F
    max_dist = int((md)/(ps2*4))
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
    target_l   = []
    patch_l    = []
    cv         = np.zeros(len(grid)) 
    dist       = np.zeros(len(grid))
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
        target = edges0F[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]*circle1
        if target.shape != (2*w,2*w):
            continue
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
        dist[i] = META.calc_distance(lat,lon,lat_0,lon_0)
        dist_lon[i] = lon_0-lon
        dist_lat[i] = lat_0-lat
        target_l.append(target)
        if edges1Fa[x_i-w:x_i+w,y_i-w:y_i+w].shape != (2*w,2*w):
            patch_l.append(circle1)
        else:
            patch_l.append(edges1Fa[x_i-w:x_i+w,y_i-w:y_i+w]*circle1)  
        o_x[i] = x_i
        o_y[i] = y_i
        t_x[i] = x_i_0
        t_y[i] = y_i_0
        # For referencing:
        origin_x[i] = x_i*fx_F
        origin_y[i] = y_i*fy_F
        target_lon[i] = lon_0
        target_lat[i] = lat_0
    return dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, RECC_over, target_l, patch_l, cv

def remove_outliers(ps2,dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv):
    size0 = len(dist)
    indices = np.where(cv>0)[0]
    dist       = dist[indices]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    o_x        = o_x[indices]
    o_y        = o_y[indices]
    t_x        = t_x[indices]
    t_y        = t_y[indices]
    cv         = cv[indices]
    size1=len(dist)
    if len(dist[cv<1.5]) >= 50:
        ind = np.where(cv<1.5)[0]
    elif len(dist[cv<4]) >= 50:
        ind = np.where(cv<4)[0]
    else:
        ind = np.where(cv<np.median(cv))[0]
        print("WARNING   : Not enough points with low CV score.")
    sub_d_x = t_x[ind]-o_x[ind]
    sub_d_y = t_y[ind]-o_y[ind]
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
    delta_x = (t_x - o_x) - x_offset
    delta_y = (t_y - o_y) - y_offset
    distance = delta_x**2 + delta_y**2
    radius = (2/ps2)**2
    indices = np.where(distance <= radius)[0]
    dist       = dist[indices]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    o_x        = o_x[indices]
    o_y        = o_y[indices]
    t_x        = t_x[indices]
    t_y        = t_y[indices]
    cv         = cv[indices]
    size2=len(dist)  
    print("GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")   
    gcplist = []
    for k in range(len(origin_x)): 
        gcplist.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    return gcplist, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv
      
def georeference(wdir,path0,file,gcplist):
    pbar3 = tqdm(total=1,position=0,desc="Georeg    ")
    path1 = wdir+"\\"+file+"_geodep.vrt"
    path2 = wdir+"\\"+file+"_georeg.vrt"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    #os.system("gdal_translate -a_srs EPSG:4326 -of VRT"+gcplist+"\""+path0+"\" \""+path1+"\"")
    #os.system("gdalwarp -r bilinear -tps -co COMPRESS=LZW \""+path1+"\" \""+path2+"\"") 
    gdal.Translate(path1,path0,format='VRT',outputSRS= 'EPSG:4326',GCPs=gcplist)
    gdal.Warp(path2,path1,tps=True,resampleAlg='bilinear')
    pbar3.update(1)
    pbar3.close()