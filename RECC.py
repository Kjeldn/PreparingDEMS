import META
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
from random import randint
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def initgrit(mask_o,w):
    bound1 = mask_o.shape[0]
    bound2 = mask_o.shape[1]
    grid = []
    for i in range(w,bound1,w):
        for j in range(w,bound2,w):
            if mask_o[i,j] == 0:
                grid.append((i,j))
    return grid              
                
def patch_match(i, edges, gt, fact_x, fact_y, x_b, y_b, mask, edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, s, it_cancel, it_max):
    print("["+"{:.0f}".format((2+(i-1)*(ppp+3))/steps)+"%] Matching patches for image nr "+str(i)+".")
    sumedges_0 = np.zeros(edges_0.shape)
    for x in range(w,x_b_0-w):
        for y in range(w,y_b_0-w):
            sumedges_0[x,y] = sum(sum(edges_0[x-w:x+w,y-w:y+w]))
    minfeatures = np.max(sumedges_0)*0.1
    mask_0_c   = mask_0.copy()
    RECC_s     = np.zeros(edges.shape)
    dist       = np.zeros(ppp)
    dist_lon   = np.zeros(ppp)
    dist_lat   = np.zeros(ppp)
    origin_x   = np.zeros(ppp)
    origin_y   = np.zeros(ppp)
    target_lon = np.zeros(ppp)
    target_lat = np.zeros(ppp)
    o_x        = np.zeros(ppp)
    o_y        = np.zeros(ppp)
    t_x        = np.zeros(ppp)
    t_y        = np.zeros(ppp)
    j=-1
    it = 0
    if it_cancel == 1:
        while j <= ppp-2 and it*it_cancel <= it_max*ppp:
            x_i_0 = randint(w,x_b_0-w)
            y_i_0 = randint(w,y_b_0-w)
            check1 = mask_0_c[x_i_0,y_i_0]
            check2 = sumedges_0[x_i_0,y_i_0]
            if check1 <= 0 and check2 >= minfeatures:
                it = it+1
                target = edges_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
                sum_target = np.sum(target)
                RECC_s.fill(np.NaN)
                for x in range(max(w,x_i_0-s*w),min(x_b-w,x_i_0+s*w)):
                    for y in range(max(w,y_i_0-s*w),min(y_b-w,y_i_0+s*w)):
                        patch = edges[x-w:x+w,y-w:y+w]
                        RECC_s[x,y]=np.sum(np.multiply(target,patch))/(sum_target+np.sum(patch))           
                max_one  = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-1)[-1]
                max_n    = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-4-1)[-4-1]
                x_i    = np.where(RECC_s >= max_one)[1][0]
                y_i    = np.where(RECC_s >= max_one)[0][0]
                x_n      = np.where(RECC_s >= max_n)[1][0:-1]
                y_n      = np.where(RECC_s >= max_n)[0][0:-1]
                cv_score = sum(np.sqrt(np.square(x_i-x_n)+np.square(y_i-y_n)))          
                if cv_score <= cv_max:
                    lon = gt[0] + gt[1]*x_i*fact_x + gt[2]*y_i*fact_y
                    lat = gt[3] + gt[4]*x_i*fact_x + gt[5]*y_i*fact_y
                    lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_x_0 + gt_0[2]*x_i_0*fact_y_0
                    lat_0 = gt_0[3] + gt_0[4]*y_i_0*fact_x_0 + gt_0[5]*x_i_0*fact_y_0
                    dst = META.calc_distance(lat,lon,lat_0,lon_0)
                    if dst <= dst_max:
                        j=j+1
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Succesful patch-match nr "+str(j+1)+" of "+str(ppp)+".")
                        mask_0_c[x_i_0-v:x_i_0+v,y_i_0-v:y_i_0+v]=1
                        dist[j]       = dst
                        dist_lon[j]   = lon_0-lon
                        dist_lat[j]   = lat_0-lat
                        origin_x[j]   = x_i*fact_x
                        origin_y[j]   = y_i*fact_y
                        target_lon[j] = lon_0
                        target_lat[j] = lat_0
                        o_x[j] = x_i
                        o_y[j] = y_i
                        t_x[j] = x_i_0
                        t_y[j] = y_i_0
                    else:
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Match failed.")
                else:
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+",-) Match failed.")      
        dist = dist[dist!=0]
        dist_lon = dist_lon[dist_lon!=0]
        dist_lat = dist_lat[dist_lat!=0]
        origin_x = origin_x[origin_x!=0]
        origin_y = origin_y[origin_y!=0]
        target_lon = target_lon[target_lon!=0]
        target_lat = target_lat[target_lat!=0]
        o_x = o_x[o_x!=0]
        o_y = o_y[o_y!=0]
        t_x = t_x[t_x!=0]
        t_y = t_y[t_y!=0] 
        # flip the target x and y for some reason:
        t_x_temp = t_y
        t_y = t_x
        t_x = t_x_temp
    return dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y

def remove_outliers(i, ppp, conf, steps, outlier_type, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y):
    gcplist = " "
    size = len(dist)
    if outlier_type == 0:
        flier_indices = np.zeros(1)
        while len(flier_indices) >= 1:
            box_dst = plt.boxplot(dist)
            box_lon = plt.boxplot(dist_lon)
            box_lat = plt.boxplot(dist_lat)
            fliers_dst = box_dst["fliers"][0].get_data()[1]
            fliers_lon = box_lon["fliers"][0].get_data()[1]
            fliers_lat = box_lat["fliers"][0].get_data()[1]
            flier_indices_dst = np.zeros(len(fliers_dst))
            flier_indices_lon = np.zeros(len(fliers_lon))
            flier_indices_lat = np.zeros(len(fliers_lat))
            for j in range(len(fliers_dst)):
                flier_indices_dst[j] = np.where(dist==fliers_dst[j])[0][0]
            for j in range(len(fliers_lon)):
                flier_indices_lon[j] = np.where(dist_lon==fliers_lon[j])[0][0]
            for j in range(len(fliers_lat)):
                flier_indices_lat[j] = np.where(dist_lat==fliers_lat[j])[0][0]
            flier_indices = np.union1d(flier_indices_lon,flier_indices_lat)
            flier_indices = (np.union1d(flier_indices,flier_indices_dst)).astype(int)  
            dist       = np.delete(dist,flier_indices)
            dist_lon   = np.delete(dist_lon,flier_indices)
            dist_lat   = np.delete(dist_lat,flier_indices)
            origin_x   = np.delete(origin_x,flier_indices)
            origin_y   = np.delete(origin_y,flier_indices)
            target_lon = np.delete(target_lon,flier_indices)
            target_lat = np.delete(target_lat,flier_indices) 
            o_x        = np.delete(o_x,flier_indices)
            o_y        = np.delete(o_y,flier_indices)
            t_x        = np.delete(t_x,flier_indices)
            t_y        = np.delete(t_y,flier_indices)
    elif outlier_type == 1:
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
        indices = ((d_x_m/sqrt(np.var(d_x_m)))**2 + (d_y_m/sqrt(np.var(d_y_m)))**2 >= s)    
        dist       = dist[~indices]
        dist_lon   = dist_lon[~indices]
        dist_lat   = dist_lat[~indices]
        origin_x   = origin_x[~indices]
        origin_y   = origin_y[~indices]
        target_lon = target_lon[~indices]
        target_lat = target_lat[~indices]
        o_x        = o_x[~indices]
        o_y        = o_y[~indices]
        t_x        = t_x[~indices]
        t_y        = t_y[~indices]
    print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] Removed "+str(size-len(dist))+" outliers.")      
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_x[k])+" "+str(origin_y[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
    return gcplist, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y
      
def georeference(i,wdir,ppp,path,file,steps,gcplist):
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Georeferencing image nr "+str(i)+".")
    path1 = wdir+"\\temp.tif"
    path2 = wdir+"\\"+file+"_adjusted.tif"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path+"\" \""+path1+"\"")
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Succesful translate, warping...")
    os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    
    
