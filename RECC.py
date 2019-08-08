import rasterio
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from random import randint
from math import cos, sin, asin, sqrt, radians
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def initialize(wdir,files,ppp):
    path = []
    for x in range(len(files)):
        path.append(wdir+"\\"+files[x]+".tif")
    steps = (1+(len(path)-1)*(ppp+3))/100
    return(path,steps)

def calc_slope(DEM):
        gdal.DEMProcessing('slope.tif', DEM, 'slope', alg = 'ZevenbergenThorne', scale = 500000)
        with rasterio.open('slope.tif') as dataset:
            slope=dataset.read(1)
        return slope
    
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

def file_to_edges(file_type,path,luma,gamma_correct):
    if file_type == 0:
        edges, gt, fact_x, fact_y, x_b, y_b, mask = ortho_to_edges(path,luma,gamma_correct)
    elif file_type == 1:
        edges, gt, fact_x, fact_y, x_b, y_b, mask = dem_to_edges(path)
    return edges, gt, fact_x, fact_y, x_b, y_b, mask

def ortho_to_edges(path,luma,gamma_correct):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/0.6)), int(B.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_AREA)
    if gamma_correct == 1:
        Rlin                           = (R_s**2.2)/255
        Glin                           = (G_s**2.2)/255
        Blin                           = (B_s**2.2)/255
        if   luma == 709:
            Y                          = 0.2126*Rlin + 0.7152*Glin + 0.0722*Blin
        elif luma == 601:
            Y                          = 0.299*Rlin + 0.587*Glin + 0.114*Blin
        elif luma == 240:
            Y                          = 0.212*Rlin + 0.701*Glin + 0.087*Blin
        L                              = 116*Y**(1/3)-16
        arr_sg                         = ((L/np.max(L))*255).astype(np.uint8)
    elif gamma_correct == 0:
        if   luma == 709:
            arr_sg                     = 0.2126*R + 0.7152*G + 0.0722*B
        elif luma == 601:
            arr_sg                     = 0.299*R + 0.587*G + 0.114*B
        elif luma == 240:
            arr_sg                     = 0.212*R + 0.701*G + 0.087*B
    mask                               = np.zeros(arr_sg.shape)
    mask[arr_sg==255]                  = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)
    ht                                 = 250
    lt                                 = 0.5*ht
    edges                              = cv2.Canny(arr_sg,lt,ht)
    edges[mask_b>=10**-10]             = 0 
    fact_x = B.shape[0]/edges.shape[0]
    fact_y = B.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, gt, fact_x, fact_y, x_b, y_b, mask

def dem_to_edges(path):
    s                                  = 5
    file                               = gdal.Open(path)
    band                               = file.GetRasterBand(1)
    gt                                 = file.GetGeoTransform()
    arr                                = band.ReadAsArray()
    x_s, y_s                           = calc_pixsize(arr,gt)
    arr_s                              = cv2.resize(arr,(int(arr.shape[1]*(y_s/0.5)), int(arr.shape[0]*(x_s/0.5))),interpolation = cv2.INTER_AREA)
    mask                               = np.zeros(arr_s.shape)
    mask[arr_s<=0.8*np.nanmin(arr_s)]  = 1
    mask_b                             = cv2.GaussianBlur(mask,(s,s),0)
    mask_hb                            = cv2.GaussianBlur(mask,(21,21),0) 
    arr_sc                             = arr_s.copy()
    arr_sc[mask_b>=10**-10]            = np.NaN
    while np.nanmax(arr_sc)-np.nanmin(arr_sc) >= 100:
        s = s+2
        mask_b                         = cv2.GaussianBlur(mask,(s,s),0)
        arr_sc[mask_b>=10**-10]        = np.NaN
    mask_hb                            = cv2.GaussianBlur(mask,(s+6,s+6),0) 
    sort                               = np.unique(arr_sc[~np.isnan(arr_sc)])
    sortofmedian                       = sort[int(len(sort)/2)]
    std                                = np.std(arr_s[~np.isnan(arr_sc)])
    cap                                = sortofmedian + 1.5*std
    arr_scc                            = arr_sc.copy()
    arr_scc[arr_scc>=cap]              = cap
    arr_sccg                           = arr_scc.copy()
    un                                 = np.nanmin(arr_sccg)
    arr_sccg                           = arr_sccg-un
    up                                 = np.nanmax(arr_sccg)
    arr_sccg                           = (arr_sccg/up)*255
    arr_sccgb                          = cv2.GaussianBlur(arr_sccg,(3,3),0)
    arr_sccgbf                         = arr_sccgb.copy()
    arr_sccgbf[np.isnan(arr_sccgbf)]   = np.nanmean(arr_sccgbf)
    arr_sccgbf                         = arr_sccgbf.astype(np.uint8)    
    arr_sccgbfa                        = cv2.adaptiveThreshold(arr_sccgbf,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
    arr_sccgbfa[mask_hb>=10**-10]      = 0
    edges                              = cv2.medianBlur(arr_sccgbfa,5) 
    fact_x = arr.shape[0]/edges.shape[0]
    fact_y = arr.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, gt, fact_x, fact_y, x_b, y_b, mask
                
def patch_match(i, edges, gt, fact_x, fact_y, x_b, y_b, mask, edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, it_cancel, it_max):
    sumedges_0 = np.zeros(edges_0.shape)
    for x in range(w,x_b_0-w):
        for y in range(w,y_b_0-w):
            sumedges_0[x,y] = sum(sum(edges_0[x-w:x+w,y-w:y+w]))
    minfeatures = np.max(sumedges_0)*0.1
    mask_0_c   = mask_0.copy()
    dist       = np.zeros(ppp)
    dist_lon   = np.zeros(ppp)
    dist_lat   = np.zeros(ppp)
    origin_x   = np.zeros(ppp)
    origin_y   = np.zeros(ppp)
    target_lon = np.zeros(ppp)
    target_lat = np.zeros(ppp)
    j=-1
    it = 0
    if it_cancel == 1:
        while j <= ppp-2 and it <= 10*ppp:
            x_i_0 = randint(w,x_b_0-w)
            y_i_0 = randint(w,y_b_0-w)
            check1 = mask_0_c[x_i_0,y_i_0]
            check2 = sumedges_0[x_i_0,y_i_0]
            if check1 <= 0 and check2 >= minfeatures:
                it = it+1
                target = edges_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
                sum_target = np.sum(target)
                RECC_s = np.zeros(edges.shape)
                for x in range(max(w,x_i_0-4*w),min(x_b-w,x_i_0+4*w)):
                    for y in range(max(w,y_i_0-4*w),min(y_b-w,y_i_0+4*w)):
                        patch = edges[x-w:x+w,y-w:y+w]
                        RECC_s[x,y]=np.sum(np.multiply(target,patch))/(sum_target+np.sum(patch))           
                max_one  = np.partition(RECC_s.flatten(),-1)[-1]
                max_n    = np.partition(RECC_s.flatten(),-4-1)[-4-1]
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
                    dst = calc_distance(lat,lon,lat_0,lon_0)
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
                    else:
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Match failed.")
                else:
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+",-) Match failed.")
                if it == it_max*ppp+1:
                    dist = dist[dist!=0]
                    dist_lon = dist_lon[dist_lon!=0]
                    dist_lat = dist_lat[dist_lat!=0]
                    origin_x = origin_x[origin_x!=0]
                    origin_y = origin_y[origin_y!=0]
                    target_lon = target_lon[target_lon!=0]
                    target_lat = target_lat[target_lat!=0]
        if it_cancel == 0:
        while j <= ppp-2:
            x_i_0 = randint(w,x_b_0-w)
            y_i_0 = randint(w,y_b_0-w)
            check1 = mask_0_c[x_i_0,y_i_0]
            check2 = sumedges_0[x_i_0,y_i_0]
            if check1 <= 0 and check2 >= minfeatures:
                it = it+1
                target = edges_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
                sum_target = np.sum(target)
                RECC_s = np.zeros(edges.shape)
                for x in range(max(w,x_i_0-4*w),min(x_b-w,x_i_0+4*w)):
                    for y in range(max(w,y_i_0-4*w),min(y_b-w,y_i_0+4*w)):
                        patch = edges[x-w:x+w,y-w:y+w]
                        RECC_s[x,y]=np.sum(np.multiply(target,patch))/(sum_target+np.sum(patch))           
                max_one  = np.partition(RECC_s.flatten(),-1)[-1]
                max_n    = np.partition(RECC_s.flatten(),-4-1)[-4-1]
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
                    dst = calc_distance(lat,lon,lat_0,lon_0)
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
                    else:
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Match failed.")
                else:
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+",-) Match failed.")
    return dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat

def remove_outliers(i, ppp, steps, outlier_type, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat):
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
    elif outlier_type == 1:        # OUTDATED:
        mean = np.mean(dist)
        std = np.std(dist)
        bol1 = np.linspace(0,len(dist)-1,len(dist))
        bol2 = np.zeros(dist.shape)
        bol2[dist<=mean-2.58*(std/sqrt(len(dist)))]=1
        bol2[dist>=mean+2.58*(std/sqrt(len(dist)))]=1
        bol = (bol1*bol2).astype(int)
        dist = np.delete(dist,bol)
        dist_lon   = np.delete(dist_lon,bol)
        dist_lat   = np.delete(dist_lat,bol)
        origin_x   = np.delete(origin_x,bol)
        origin_y   = np.delete(origin_y,bol)
        target_lon = np.delete(target_lon,bol)
        target_lat = np.delete(target_lat,bol)
    print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] Removed "+str(size-len(dist))+" outliers.")      
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_x[k])+" "+str(origin_y[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
    return gcplist, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat
    
    
def georeference(wdir,path,file,gcplist):
    path1 = wdir+"\\temp.tif"
    path2 = wdir+"\\"+file+"_adjusted.tif"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path+"\" \""+path1+"\"")
    os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    
