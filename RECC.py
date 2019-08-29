import META
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm

                
def patch_match(pixel_size, w, s, dst_max, edges1C, gt, fact_x, fact_y, x_b, y_b, edges0C, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_o_0):
    max_dist = int((dst_max)/pixel_size)
    buffer = 20
    bound1 = mask_o_0.shape[0]
    bound2 = mask_o_0.shape[1]
    grid = []
    for i in range(max_dist+w+buffer,bound1-max_dist-w-buffer,s):
        for j in range(max_dist+w+buffer,bound2-max_dist-w-buffer,s):
            if mask_o_0[i,j] == 0:
                grid.append((i,j))
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
    RECC_total = np.zeros(edges1C.shape)
    RECC_over  = np.zeros(edges1C.shape)
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
        sum_target = np.sum(target)   
        lat_0 = gt_0[3] + gt_0[5]*x_i_0*fact_x_0  
        lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_y_0
        x_i_0_og = int((lat_0-gt[3])/(gt[5]*fact_x))
        y_i_0_og = int((lon_0-gt[0])/(gt[1]*fact_y)) 
        distance = np.sqrt((x_i_0-x_i_0_og)**2+(y_i_0-y_i_0_og)**2)
        search_wide = edges1C[x_i_0_og-max_dist-w:x_i_0_og+max_dist+w,y_i_0_og-max_dist-w:y_i_0_og+max_dist+w]
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

def remove_outliers(conf, cv_thresh, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv):
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
    indices    = np.where(cv<=cv_thresh)[0]
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
    if   conf == 95:
        s = 5.991
    elif conf == 90:
        s = 4.605
    elif conf == 80:
        s = 3.219
    elif conf == 75:
        s = 2.770
    elif conf == 50:
        s = 1.388
    d_x = t_x - o_x
    d_y = t_y - o_y
    d_x_m = d_x - np.median(d_x)
    d_y_m = d_y - np.median(d_y)        
    indices = ((d_x_m/sqrt(np.var(d_x_m)))**2 + (d_y_m/sqrt(np.var(d_y_m)))**2 <= s)    
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
    size3=len(dist)
    print("GCP status: ("+str(size3)+"/"+str(size0-size1)+"/"+str(size1-size2)+"/"+str(size2-size3)+") [OK/OoD/CV/2D]")   
    gcplist = " "
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_y[k])+" "+str(origin_x[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
    return gcplist, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv
      
def georeference(wdir,path,file,gcplist):
    if gcplist.count('gcp') <= 20:
        print("Not enough GCPs for georegistration.")
    else:
        pbar3 = tqdm(total=2,position=0,desc="Georegist ")
        path1 = wdir+"\\temp.tif"
        path2 = wdir+"\\"+file+"_adjusted.tif"
        if os.path.isfile(path1.replace("\\","/")):
            os.remove(path1)
        if os.path.isfile(path2.replace("\\","/")):
            os.remove(path2)
        os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path+"\" \""+path1+"\"")
        pbar3.update(1)
        os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    
        pbar3.update(1)
        pbar3.close()