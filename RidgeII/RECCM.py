import METAA
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
from shapely.ops import triangulate
from scipy.interpolate import interp1d
from tempfile import mkstemp
from shutil import move
from mpl_toolkits.mplot3d import Axes3D

def SinglMatch(plist,Edges1C,gt1C,fx1C,fy1C,Edges0C,gt0C,fx0C,fy0C,MaskB0C):
    psC = 0.5
    md = 12
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
    lat_0 = gt0C[3] + gt0C[5]*x0*fx0C  
    lon_0 = gt0C[0] + gt0C[1]*y0*fy0C
    xog = int(round((lat_0-gt1C[3])/(gt1C[5]*fx1C)))
    yog = int(round((lon_0-gt1C[0])/(gt1C[1]*fy1C)))     
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
    print("Status    : ("+str(x_off)+"m,"+str(y_off)+"m), CV: "+str(round(CV1,2)))  
    return plist,x_off,y_off,x0,y0,xog,yog,x1,y1,CV1

def PatchMatch(plist,Edges1F, gt1F, fx1F, fy1F, Edges0F, gt0F, fx0F, fy0F, MaskB0F,x_off,y_off,CV1):
    ps0F = 0.05
    w = int(25/ps0F)
    buffer = 2*w
    edges1Fa = np.zeros((Edges1F.shape[0]+buffer*2,Edges1F.shape[1]+2*buffer))
    edges1Fa[buffer:-buffer,buffer:-buffer] = Edges1F    
    if CV1>4:
        md = 10
    elif CV1<1.5:
        md = 5
    else:
        md = 5 + 5*((CV1-1.5)/2.5)
    max_dist = int((md)/(ps0F))
    contours,hierarchy = cv2.findContours((1-MaskB0F).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    polygon = Polygon(np.array(biggest_contour[:,0]))
    ref_area = polygon.area
    polygon = polygon.buffer(-w)
    v=w
    while polygon.is_empty or polygon.geom_type == 'MultiPolygon' or polygon.area<0.4*ref_area:
        v -= int(2/ps0F)
        polygon = Polygon(np.array(biggest_contour[:,0]))
        polygon = polygon.buffer(-v)
    if v != w:
        print("WARNING   : Polygon-buffer: "+str(v*ps0F)+" < 25...")
    x,y = polygon.exterior.xy   
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]
    fx, fy = interp1d( distance, x ), interp1d( distance, y )
    alpha = np.linspace(0, 1, 200)
    x_regular, y_regular = fx(alpha), fy(alpha)
    grid = []
    for i in range(len(x_regular)):
        grid.append((int(round(x_regular[i])),int(round(y_regular[i])))) 
    if polygon.buffer(-2*w).is_empty == False:
        polygon = polygon.buffer(-2*w)
        x,y = polygon.exterior.xy   
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        alpha = np.linspace(0, 1, 100)
        x_regular, y_regular = fx(alpha), fy(alpha)
        for i in range(len(x_regular)):
            grid.append((int(round(x_regular[i])),int(round(y_regular[i])))) 
    CVa        = np.zeros(len(grid)) 
    x0         = np.zeros(len(grid)).astype(int)
    y0         = np.zeros(len(grid)).astype(int)
    xog        = np.zeros(len(grid)).astype(int)
    yog        = np.zeros(len(grid)).astype(int)
    xof        = np.zeros(len(grid)).astype(int)
    yof        = np.zeros(len(grid)).astype(int)
    x1         = np.zeros(len(grid)).astype(int)
    y1         = np.zeros(len(grid)).astype(int)
    origin_x   = np.zeros(len(grid))
    origin_y   = np.zeros(len(grid))
    target_lon = np.zeros(len(grid))
    target_lat = np.zeros(len(grid))
    RECC_total = np.zeros(Edges1F.shape)
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
    for i in tqdm(range(len(grid)),position=0,miniters=int(len(grid)/10),desc="RECC(f)   "):
        x0[i] = grid[i][1]
        y0[i] = grid[i][0]
        target_lon[i] = gt0F[0] + gt0F[1]*y0[i]*fy0F
        target_lat[i] = gt0F[3] + gt0F[5]*x0[i]*fx0F  
        xog[i] = int(round((target_lat[i]-gt1F[3])/(gt1F[5]*fx1F)))
        yog[i] = int(round((target_lon[i]-gt1F[0])/(gt1F[1]*fy1F)))
        xof[i] = int(round(xog[i] + x_off/ps0F))
        yof[i] = int(round(yog[i] + y_off/ps0F))
        target = Edges0F[x0[i]-w:x0[i]+w,y0[i]-w:y0[i]+w]
        if target.shape != (2*w,2*w):
            continue
        target = target*circle1
        sum_target = np.sum(target)     
        search_wide = np.zeros((2*(max_dist+w),2*(max_dist+w)))
        search_wide = edges1Fa[buffer+xof[i]-max_dist-w:buffer+xof[i]+max_dist+w,buffer+yof[i]-max_dist-w:buffer+yof[i]+max_dist+w]
        if search_wide.shape != (2*(max_dist+w),2*(max_dist+w)):
            continue
        sum_patch = cv2.filter2D(search_wide,-1,circle1)
        numerator = cv2.filter2D(search_wide,-1,target)
        RECC_wide = numerator / (sum_patch+sum_target)
        RECC_area = RECC_wide[w:-w,w:-w]*circle2
        RECC_total.fill(np.NaN)
        if RECC_total[xof[i]-max_dist:xof[i]+max_dist,yof[i]-max_dist:yof[i]+max_dist].shape != (2*(max_dist),2*(max_dist)):
            continue
        RECC_total[xof[i]-max_dist:xof[i]+max_dist,yof[i]-max_dist:yof[i]+max_dist] = RECC_area
        max_one  = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-1)[-1]
        max_n    = np.partition(RECC_total[~np.isnan(RECC_total)].flatten(),-4-1)[-4-1]
        y1[i]    = np.where(RECC_total >= max_one)[1][0]  
        x1[i]    = np.where(RECC_total >= max_one)[0][0]
        y_n      = np.where(RECC_total >= max_n)[1][0:-1]
        x_n      = np.where(RECC_total >= max_n)[0][0:-1]
        CVa[i] = sum(np.sqrt(np.square(x1[i]-x_n)+np.square(y1[i]-y_n)))/4
        origin_x[i] = x1[i]*fx1F
        origin_y[i] = y1[i]*fy1F    
    dx = (x1-xof)*ps0F
    dy = (y1-yof)*ps0F 
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
    plt.figure(257)
    plt.subplot(1,2,1)
    plt.imshow(Edges0F,cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow(Edges1F,cmap='Greys')
    plt.figure(258)
    plt.subplot(1,2,1)
    plt.imshow(Edges0F,cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow(Edges1F,cmap='Greys')
    return plist,origin_x,origin_y,target_lon,target_lat,x0,y0,xog,yog,xof,yof,x1,y1,CVa,dx,dy

def RemOutlier(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,files,iiii):
    size0 = len(x0)
    indices = np.where(CVa>0)[0]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CVa       = CVa[indices]
    dx        = dx[indices]
    dy        = dy[indices]
    size1=len(x0)
    clist = list(np.random.choice(range(256), size=len(x0)))
    p=plt.figure(257)
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title(str(size1)+" in-domain")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close(257)
    plist.append(p)
    clist = np.array(clist)
    if len(x0[CVa<1.5]) >= 50:
        ind = np.where(CVa<1.5)[0]
    elif len(x0[CVa<4]) >= 50:
        ind = np.where(CVa<4)[0]
    else:
        ind = np.where(CVa<np.median(CVa))[0]
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
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CVa       = CVa[indices]
    clist     = clist[indices]
    size2=len(x0)  
    print("GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]") 
    gcplist_DEM = []
    for k in range(len(origin_x)): 
        gcplist_DEM.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    gto = gdal.Open(files[iiii]).GetGeoTransform()
    origin_x = ((gt1F[3]+gt1F[5]*origin_x) - gto[3])/gto[5]
    origin_y = ((gt1F[0]+gt1F[1]*origin_y) - gto[0])/gto[1]
    gcplist = []
    for k in range(len(origin_x)): 
        gcplist.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    clist = list(clist)
    p = plt.figure(258)
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title("("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close(258)
    plist.append(p)
    return plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist,gcplist_DEM

# RECCM.RemOutSlop(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,path,i)

def RemOutSlop(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,files,iiii):
    size0 = len(x0)
    indices = np.where(CVa>0)[0]
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CVa       = CVa[indices]
    dx        = dx[indices]
    dy        = dy[indices]
    size1=len(x0)
    clist = list(np.random.choice(range(256), size=len(x0)))
    p=plt.figure(257)
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title(str(size1)+" in-domain")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close(257)
    plist.append(p)
    clist = np.array(clist)
    if len(x0[CVa<1.5]) >= size0/2:
        ind = np.where(CVa<1.5)[0]
    elif len(x0[CVa<4]) >= size0/2:
        ind = np.where(CVa<4)[0]
    else:
        ind = np.where(CVa<np.median(CVa))[0]
        print("WARNING   : Not enough points with low CV score.")
    fun_dx = METAA.fit(origin_x[ind],origin_y[ind],dx[ind])
    fun_dy = METAA.fit(origin_x[ind],origin_y[ind],dy[ind])
    supposed_dx = fun_dx[0]*origin_x+fun_dx[1]*origin_y+fun_dx[2]
    supposed_dy = fun_dy[0]*origin_x+fun_dy[1]*origin_y+fun_dy[2]
    delta_x = dx - supposed_dx
    delta_y = dy - supposed_dy
    distance = np.square(delta_x) + np.square(delta_y)
    radius = 0.1
    indices = np.where(distance.T <= radius)[0]
    inv_ind = []
    for i in range(len(origin_x)):
        if i not in ind:
            inv_ind.append(i)
    p = plt.figure()
    ax = p.add_subplot(111, projection='3d')
    ax.scatter(origin_x,origin_y,supposed_dx,c='b',marker='o')
    ax.scatter(origin_x[ind],origin_y[ind],dx[ind],c='g',marker='o')
    ax.scatter(origin_x[inv_ind],origin_y[inv_ind],dx[inv_ind],c='r',marker='o')
    ax.set_zlim(min(dx)-0.05, max(dx)+0.05)
    plt.close()
    plist.append(p)   
    p = plt.figure()
    ax = p.add_subplot(111, projection='3d')
    ax.scatter(origin_x,origin_y,supposed_dy,c='b',marker='o')
    ax.scatter(origin_x[ind],origin_y[ind],dy[ind],c='g',marker='o')
    ax.scatter(origin_x[inv_ind],origin_y[inv_ind],dy[inv_ind],c='r',marker='o')
    ax.set_zlim(min(dy)-0.05, max(dy)+0.05)
    plt.close()
    plist.append(p)  
    origin_x   = origin_x[indices]
    origin_y   = origin_y[indices]
    target_lon = target_lon[indices]
    target_lat = target_lat[indices]
    x0        = x0[indices]
    y0        = y0[indices]
    x1        = x1[indices]
    y1        = y1[indices]
    CVa       = CVa[indices]
    clist     = clist[indices]
    size2=len(x0)  
    print("GCP status: ("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]") 
    gcplist_DEM = []
    for k in range(len(origin_x)): 
        gcplist_DEM.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    gto = gdal.Open(files[iiii]).GetGeoTransform()
    origin_x = ((gt1F[3]+gt1F[5]*origin_x) - gto[3])/gto[5]
    origin_y = ((gt1F[0]+gt1F[1]*origin_y) - gto[0])/gto[1]
    gcplist = []
    for k in range(len(origin_x)): 
        gcplist.append(gdal.GCP(target_lon[k],target_lat[k],0,origin_y[k],origin_x[k]))
    clist = list(clist)
    p = plt.figure(258)
    plt.subplot(1,2,1)
    plt.title("GCP Status:")
    plt.scatter(y0,x0,s=1,c=clist)
    plt.subplot(1,2,2)
    plt.title("("+str(size2)+"/"+str(size0-size1)+"/"+str(size1-size2)+") [OK/OoD/CV-2D]")
    plt.scatter(y1,x1,s=1,c=clist)
    plt.close(258)
    plist.append(p)
    return plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist,gcplist_DEM

def Georegistr(i,files,gcplist,gcplist_DEM):
    pbar3 = tqdm(total=2,position=0,desc="Georeg    ")
    temp = files[i][::-1]
    temp2 = temp[:temp.find("/")]
    src = temp2[::-1]
    dest = files[i].strip(".tif")+"_GR.vrt"  
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
   
    file = files[i].strip(".tif")+"_DEM.tif"
    temp = file[::-1]
    temp2 = temp[:temp.find("/")]
    src = temp2[::-1]
    dest = file.strip(".tif")+"_GR.vrt"  
    if os.path.isfile(dest.replace("\\","/")):
        os.remove(dest)
    temp = gdal.Translate('',file,format='VRT',outputSRS= 'EPSG:4326',GCPs=gcplist_DEM)
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
    
def GeoPointss(i,files,target_lon,target_lat,origin_x,origin_y,gt1F):
    pbar3 = tqdm(total=1,position=0,desc="GeoPoints ")
    origin_lat = gt1F[3]+gt1F[5]*origin_x
    origin_lon = gt1F[0]+gt1F[1]*origin_y
    dest = files[i].strip(".tif")+".points" 
    if os.path.isfile(dest.replace("\\","/")):
        os.remove(dest)
    f = open(dest,"w+")
    f.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual")
    for i in range(len(target_lon)):
        f.write("\n"+str(target_lon[i])+","+str(target_lat[i])+","+str(origin_lon[i])+","+str(origin_lat[i])+",1,0,0,0")
    f.close()
    pbar3.update(1)
    pbar3.close()