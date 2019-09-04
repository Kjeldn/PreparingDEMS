import META
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from random import randint
import math
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
                
def patch_match(pixel_size, w, dst_max, edges1C, gt, fact_x, fact_y, x_b, y_b, edges0C, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_o_0):
    buffer = 2*w
    edges1Ca = np.zeros((edges1C.shape[0]+buffer*2,edges1C.shape[1]+2*buffer))
    edges1Ca[buffer:-buffer,buffer:-buffer] = edges1C
    pixel_size=0.05
    dst_max=12
    max_dist = int((dst_max)/pixel_size)
    xlist = np.where(mask_o_0==0)[0]
    ylist = np.where(mask_o_0==0)[1]
    ind1 = np.where(xlist==max(xlist))[0][-1]
    ind2 = np.where(ylist==max(ylist))[0][0]
    ind3 = np.where(xlist==min(xlist))[0][0]
    ind4 = np.where(ylist==min(ylist))[0][-1]
    polygon = Polygon([(xlist[ind1],ylist[ind1]),(xlist[ind2],ylist[ind2]),(xlist[ind3],ylist[ind3]),(xlist[ind4],ylist[ind4])])
    polygon = polygon.buffer(-sqrt(w**2+w**2))
    v=w
    while polygon.is_empty or len(polygon.exterior.xy[0]) <= 4:
        v -= 50
        polygon = Polygon([(xlist[ind1],ylist[ind1]),(xlist[ind2],ylist[ind2]),(xlist[ind3],ylist[ind3]),(xlist[ind4],ylist[ind4])])
        polygon = polygon.buffer(-sqrt(v**2+v**2))
    if v != w:
        print("WARNING   : Polygon-buffer ("+str(v)+") < w...")
    c_x,c_y = polygon.exterior.xy
    c_x = [int(round(x)) for x in c_x]
    c_y = [int(round(y)) for y in c_y] 
    dist = max((max(c_x)-min(c_x)),(max(c_y)-min(c_y)))
    s = int(min(1000,dist/10))
    grid = []
    while len(grid) <= 100 and s > w/100:
        grid=[]
        for i in range(len(c_x)-1):
            j=i+1
            if c_x[i] > c_x[j]:
                t = i; i = j; j = t
            xdiff = abs(c_x[i]-c_x[j])
            s_a = int(xdiff/(math.ceil(xdiff/s))) 
            m,b = np.polyfit((c_x[i],c_x[j]),(c_y[i],c_y[j]),1) 
            for xval in range(c_x[i],c_x[j],s_a):
                yval = b+m*xval
                xval = int(round(xval))
                yval = int(round(yval))
                grid.append((xval,yval))
        s -= 50
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
    RECC_total = np.zeros(edges1Ca.shape)
    RECC_over  = np.zeros(edges1Ca.shape)
    RECC_over.fill(np.NaN)
    circle = np.zeros((2*max_dist,2*max_dist))
    for x in range(circle.shape[0]):
        for y in range(circle.shape[1]):
            if (x-max_dist)**2 + (y-max_dist)**2 < max_dist**2:
                circle[x,y]=1
    circle[circle==0]=np.NaN
    for i in tqdm(range(len(grid)),position=0,desc="RECC      "):
        x_i_0 = grid[i][0]
        y_i_0 = grid[i][1]
        target = edges0C[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
        if target.shape != (2*w,2*w):
            continue
        sum_target = np.sum(target)   
        lat_0 = gt_0[3] + gt_0[5]*x_i_0*fact_x_0  
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_y_0
        x_i_0_og = int(round((lat_0-gt[3])/(gt[5]*fact_x)))
        y_i_0_og = int(round((lon_0-gt[0])/(gt[1]*fact_y))) 
        search_wide = np.zeros((2*(max_dist+w),2*(max_dist+w)))
        search_wide = edges1Ca[buffer+x_i_0_og-max_dist-w:buffer+x_i_0_og+max_dist+w,buffer+y_i_0_og-max_dist-w:buffer+y_i_0_og+max_dist+w]
        if search_wide.shape != (2*(max_dist+w),2*(max_dist+w)):
            continue
        sum_patch = cv2.filter2D(search_wide,-1,np.ones(target.shape))
        numerator = cv2.filter2D(search_wide,-1,target)
        RECC_wide = numerator / (sum_patch+sum_target)
        RECC_area = RECC_wide[w:-w,w:-w]*circle
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
        lon = gt[0] + gt[1]*y_i*fact_y + gt[2]*x_i*fact_x
        lat = gt[3] + gt[4]*y_i*fact_y + gt[5]*x_i*fact_x
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_y_0 + gt_0[2]*x_i_0*fact_x_0
        lat_0 = gt_0[3] + gt_0[4]*y_i_0*fact_y_0 + gt_0[5]*x_i_0*fact_x_0
        dist[i] = META.calc_distance(lat,lon,lat_0,lon_0)
        dist_lon[i] = lon_0-lon
        dist_lat[i] = lat_0-lat
        target_l.append(target)
        patch_l.append(edges1C[x_i-w:x_i+w,y_i-w:y_i+w])  
        o_x[i] = x_i
        o_y[i] = y_i
        t_x[i] = x_i_0
        t_y[i] = y_i_0
        # For referencing:
        origin_x[i] = x_i*fact_x
        origin_y[i] = y_i*fact_y
        target_lon[i] = lon_0
        target_lat[i] = lat_0
    return dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, RECC_over, target_l, patch_l, cv

#def remove_outliers(conf, cv_thresh, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv):
#    size0 = len(dist)
#    indices = np.where(cv>0)[0]
#    dist       = dist[indices]
#    origin_x   = origin_x[indices]
#    origin_y   = origin_y[indices]
#    target_lon = target_lon[indices]
#    target_lat = target_lat[indices]
#    o_x        = o_x[indices]
#    o_y        = o_y[indices]
#    t_x        = t_x[indices]
#    t_y        = t_y[indices]
#    cv         = cv[indices]
#    size1=len(dist)
#    indices     = np.where(cv<=cv_thresh)[0]
#    Cdist       = dist[indices]
#    Corigin_x   = origin_x[indices]
#    Corigin_y   = origin_y[indices]
#    Ctarget_lon = target_lon[indices]
#    Ctarget_lat = target_lat[indices]
#    Co_x        = o_x[indices]
#    Co_y        = o_y[indices]
#    Ct_x        = t_x[indices]
#    Ct_y        = t_y[indices]
#    Ccv         = cv[indices]
#    size2=len(Cdist)
#    if   conf == 95:
#        s = 5.991
#    elif conf == 90:
#        s = 4.605
#    elif conf == 80:
#        s = 3.219
#    elif conf == 75:
#        s = 2.770
#    elif conf == 50:
#        s = 1.388
#    Cd_x = Ct_x - Co_x
#    Cd_y = Ct_y - Co_y
#    Cd_x_m = Cd_x - np.median(Cd_x)
#    Cd_y_m = Cd_y - np.median(Cd_y)        
#    indices = ((Cd_x_m/sqrt(np.var(Cd_x_m)))**2 + (Cd_y_m/sqrt(np.var(Cd_y_m)))**2 <= s)   
#    if len(indices) >= 20:
#        dist       = Cdist[indices]
#        origin_x   = Corigin_x[indices]
#        origin_y   = Corigin_y[indices]
#        target_lon = Ctarget_lon[indices]
#        target_lat = Ctarget_lat[indices]
#        o_x        = Co_x[indices]
#        o_y        = Co_y[indices]
#        t_x        = Ct_x[indices]
#        t_y        = Ct_y[indices]
#        cv         = Ccv[indices]
#    else:
#        print("WARNING   : Used reverse 2D confidence interval...")
#        d_x = t_x - o_x
#        d_y = t_y - o_y
#        d_x_m = d_x - np.median(Cd_x)
#        d_y_m = d_y - np.median(Cd_y)     
#        indices = ((d_x_m/sqrt(np.var(d_x_m)))**2 + (d_y_m/sqrt(np.var(d_y_m)))**2 <= s) 
#        dist       = dist[indices]
#        origin_x   = origin_x[indices]
#        origin_y   = origin_y[indices]
#        target_lon = target_lon[indices]
#        target_lat = target_lat[indices]
#        o_x        = o_x[indices]
#        o_y        = o_y[indices]
#        t_x        = t_x[indices]
#        t_y        = t_y[indices]
#        cv         = cv[indices]
#    size3=len(dist)
#    s=3.219
#    d_x = t_x-o_x
#    d_y = t_y-o_y
#    d_x_m = d_x-np.median(d_x)
#    d_y_m = d_y-np.median(d_y)
#    indices = ((d_x_m/sqrt(np.var(d_x_m)))**2 + (d_y_m/sqrt(np.var(d_y_m)))**2 <= s) 
#    dist       = dist[indices]
#    origin_x   = origin_x[indices]
#    origin_y   = origin_y[indices]
#    target_lon = target_lon[indices]
#    target_lat = target_lat[indices]
#    o_x        = o_x[indices]
#    o_y        = o_y[indices]
#    t_x        = t_x[indices]
#    t_y        = t_y[indices]
#    cv         = cv[indices]
#    size4=len(dist)
#    print("GCP status: ("+str(size4)+"/"+str(size0-size1)+"/"+str(size1-size2)+"/"+str(size2-size3)+"/"+str(size3-size4)+") [OK/OoD/CV/2D/2D]")   
#    gcplist = " "
#    for k in range(len(origin_x)):
#        gcplist = gcplist+"-gcp "+str(origin_y[k])+" "+str(origin_x[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
#    return gcplist, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv

def remove_outliers2(ps2,dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv):
    flag = 0
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
    indices     = np.where(cv<=4)[0]
    Cdist       = dist[indices]
    Corigin_x   = origin_x[indices]
    Corigin_y   = origin_y[indices]
    Ctarget_lon = target_lon[indices]
    Ctarget_lat = target_lat[indices]
    Co_x        = o_x[indices]
    Co_y        = o_y[indices]
    Ct_x        = t_x[indices]
    Ct_y        = t_y[indices]
    Ccv         = cv[indices]
    size2=len(Cdist)
    slist = list([1.388,2.770,3.219,4.605,5.991])
    for s in slist:
        Cd_x = Ct_x - Co_x
        Cd_y = Ct_y - Co_y
        Cd_x_m = Cd_x - np.median(Cd_x)
        Cd_y_m = Cd_y - np.median(Cd_y)        
        indices = ((Cd_x_m/sqrt(np.var(Cd_x_m)))**2 + (Cd_y_m/sqrt(np.var(Cd_y_m)))**2 <= s)   
        Cdist       = Cdist[indices]
        Corigin_x   = Corigin_x[indices]
        Corigin_y   = Corigin_y[indices]
        Ctarget_lon = Ctarget_lon[indices]
        Ctarget_lat = Ctarget_lat[indices]
        Co_x        = Co_x[indices]
        Co_y        = Co_y[indices]
        Ct_x        = Ct_x[indices]
        Ct_y        = Ct_y[indices]
        Ccv         = Ccv[indices]
        if abs(np.sum(Cdist-np.median(Cdist)))/len(Cdist) < 0.1:
            flag = 1
            break
    if len(Cdist) <= 20:
        print("WARNING   : Reverse 2D Confidence Interval applied...")
        flag = 0
        Cdist       = dist
        Corigin_x   = origin_x
        Corigin_y   = origin_y
        Ctarget_lon = target_lon
        Ctarget_lat = target_lat
        Co_x        = o_x
        Co_y        = o_y
        Ct_x        = t_x
        Ct_y        = t_y
        Ccv         = cv
        size2=len(Cdist)
        for s in slist:
            Cd_xcv = Ct_x[np.where(Ccv<=4)[0]] - Co_x[np.where(Ccv<=4)[0]]
            Cd_ycv = Ct_y[np.where(Ccv<=4)[0]] - Co_y[np.where(Ccv<=4)[0]]
            Cd_x = Ct_x - Co_x
            Cd_y = Ct_y - Co_y
            Cd_x_m = Cd_x - np.median(Cd_xcv)
            Cd_y_m = Cd_y - np.median(Cd_ycv)
            indices = ((Cd_x_m/sqrt(np.var(Cd_x_m)))**2 + (Cd_y_m/sqrt(np.var(Cd_y_m)))**2 <= s)   
            Cdist       = Cdist[indices]
            Corigin_x   = Corigin_x[indices]
            Corigin_y   = Corigin_y[indices]
            Ctarget_lon = Ctarget_lon[indices]
            Ctarget_lat = Ctarget_lat[indices]
            Co_x        = Co_x[indices]
            Co_y        = Co_y[indices]
            Ct_x        = Ct_x[indices]
            Ct_y        = Ct_y[indices]
            Ccv         = Ccv[indices]
            if abs(np.sum(Cdist-np.median(Cdist)))/len(Cdist) < 0.1:
                flag = 1
                break
    dist       = Cdist
    origin_x   = Corigin_x
    origin_y   = Corigin_y
    target_lon = Ctarget_lon
    target_lat = Ctarget_lat
    o_x        = Co_x
    o_y        = Co_y
    t_x        = Ct_x
    t_y        = Ct_y
    cv         = Ccv
    size3=len(dist)
    median = np.median(dist)
    indices = []
    for i in range(len(dist)-1,-1,-1):
        if  median-ps2*2 < dist[i] < median+ps2*2:
            indices.append(i)
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
    size4=len(dist)
    print("GCP status: ("+str(size4)+"/"+str(size0-size1)+"/"+str(size1-size2)+"/"+str(size2-size3)+"/"+str(size3-size4)+"/"+str(flag)+") [OK/OoD/CV/2D/M/B]")   
    gcplist = " "
    distC = dist
    flag = 0
    while len(distC) >= 160:
        flag = 1
        origin_x = origin_x[::2]
        origin_y = origin_y[::2]
        target_lon = target_lon[::2]
        target_lat = target_lat[::2]
        distC = distC[::2]
    if flag == 1:
        print("WARNING   : Reduced GCP from "+str(len(dist))+" to "+str(len(distC)))
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_y[k])+" "+str(origin_x[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
    return gcplist, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv
      
def georeference(wdir,path,file,gcplist):
    if gcplist.count('gcp') <= 5:
        print("Not enough GCPs for georegistration.")
    else:
        pbar3 = tqdm(total=2,position=0,desc="Georeg    ")
        path1 = wdir+"\\temp.tif"
        path2 = wdir+"\\"+file+"_adjusted.tif"
        if os.path.isfile(path1.replace("\\","/")):
            os.remove(path1)
        if os.path.isfile(path2.replace("\\","/")):
            os.remove(path2)
        os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path+"\" \""+path1+"\"")
        pbar3.update(1)
        os.system("gdalwarp -r bilinear -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    
        pbar3.update(1)
        pbar3.close()