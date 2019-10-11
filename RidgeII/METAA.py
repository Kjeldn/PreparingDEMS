import os
import cv2
import gdal
import time
import shutil
import warnings
import numpy as np
import numpy.matlib
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from math import cos, sin, asin, sqrt, radians
from matplotlib.backends.backend_pdf import PdfPages

warnings.simplefilter(action = "ignore", category = RuntimeWarning)


"""
Returns multiple strings containing paths to the correct folders depending on
what PC is being used.
---
PC       | str    | Name of the PC being used
inbox    | str    | Path to inbox  
archive  | str    | Path to archive
rtu      | str    | Path to ready to upload 
nrtu     | str    | Path to not ready to upload
dstr     | str    | Path to rectified dems and points
rec      | str    | Path to receipt
"""
def GetDirs(PC):
    if PC == "STAMPERTJE":
        inbox   = r"C:\Users\VanBoven\Documents\100 Ortho Inbox"
        archive = r"D:\VanBovenDrive\VanBoven MT\Archive"
        rtu     = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\2_ready_to_upload"
        nrtu    = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\4_not_ready_to_upload"
        dstr    = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\00_rectified_DEMs_points"
        rec     = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\9_receipt"
    if PC == "Martijn":
        inbox   = r"\\STAMPERTJE\100 Ortho Inbox"
        archive = r"\\STAMPERTJE\Data\VanBovenDrive\VanBoven MT\Archive"
        rtu     = r"\\STAMPERTJE\100 Ortho Inbox\2_ready_to_upload"
        nrtu    = r"\\STAMPERTJE\100 Ortho Inbox\4_not_ready_to_upload"
        dstr    = r"\\STAMPERTJE\100 Ortho Inbox\00_rectified_DEMs_points"
        rec     = r"\\STAMPERTJE\100 Ortho Inbox\9_receipt"
    return inbox,archive,rtu,nrtu,dstr,rec 


"""
Finds orthomosaics that also have a corresponding DEM in the ortho inbox, 
consequently sorts the list of orthomosaics based on date (earliest first).
---
inbox    | str    | Path to inbox  
plist    | list   | List for figures
pathlist | list   | List with paths to suitable files in the inbox
"""
def FindFile(inbox):
    pathlist=[]
    for root, dirs, files in walk_level(inbox, level=0):
        for name in files:
            if "DEM" not in name:
                if name[-4:] == ".tif":
                    if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
                        pathlist.append(os.path.join(root,name).replace("\\","/"))
                if name[-7:] == "-GR.tif":
                    if os.path.exists(os.path.join(root,name).replace("-GR.tif","_DEM-GR.tif")) == True:
                        pathlist.append(os.path.join(root,name).replace("\\","/"))
    pathlist = sorted(pathlist, key = lambda a: filename_to_info(path_to_filename(a))[-1])
    plist = []
    plt.ioff()
    return plist,pathlist


"""
Given a path to a orthomosaic in the ortho inbox, the function checks both the
archive and the ready to upload folder for orthomosaics plus DEM pairs that
could be used for georegistration. The most recent pair is selected.
---
plist    | list   | List for figures
pathlist | list   | List with paths to suitable files in the inbox
archive  | str    | Path to archive
rtu      | str    | Path to ready to upload 
path     | str    | Path to the orthomosaic up for georegistration
base     | str    | Path to the orthomosaic that will be used as base
"""  
def FindBase(plist,pathlist,archive,rtu,path):
    company,parcel,date = filename_to_info(path_to_filename(path)) 
    candidates = walk_folder(archive,company,parcel)
    candidates_rtu = walk_folder(rtu,company,parcel)
    candidates.extend(candidates_rtu)
    dif = []
    for cand in candidates:
        dif.append(float(date)-float(filename_to_info(path_to_filename(cand))[-1]))  
    dif = np.array(dif)                                      
    if (dif<=0).all() == True:
        print("Found zero suitable orthomosaics in the past")
    dif[dif<=0]=np.NaN
    ind = np.where(dif==np.nanmin(dif))[0][:]
    flag = 0
    if len(ind) > 1:
        for i in ind:
            if candidates[i][-6:] == "GR.vrt":
                base = candidates[i]
                flag = 1
        if flag == 0:
            for i in ind:
                if candidates[i][-6:-4] == "GR":
                    base = candidates[i]
                    flag = 1
        if flag == 0:
            base = candidates[0]
    else:
        base = candidates[ind[0]]     
    flag = "File source unknown..."
    if base[0:1] == "D":
        flag = "arch"
    elif base[0:1] == "C":
        flag = "rtu"
    for i in range(len(pathlist)):
        if path == pathlist[i]:
            ind = i+1
    print("\n")
    print("BASE:", path_to_filename(base), "("+flag+")")
    print("FILE:", path_to_filename(path), "("+str(ind)+"/"+str(len(pathlist))+")")
    return plist,base


"""
Preprocessing for CannyPF and CannyLines. Takes the path to an orthomosaic,
rescales the file temporarily using GDAL, equalises the L band based on CDF,
converts it to greyscale, and blurs the image.
---
plist    | list   | List for figures
path     | str    | Path to the orthomosaic up for georegistration
img_s    | 3D arr | Rescaled and equalised orthomosaic                         (Img0C/Img1C)
img_b    | 2D arr | Rescaled, equalised, grayscaled and blurred orthomosaic    (ImgB0C/ImgB1C)
mask_b   | 2D arr | Binary map defining edges of orthomosaic                   (MaskB0C/MaskB1C)
gt       | tuple  | Geotransform corresponding to img_s, img_b and mask_b      (gt0C/gt1C)
"""
def OpenOrth(plist,path):
    pbar1 = tqdm(total=1,position=0,desc="OrtOpening")
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()                
    x_s, y_s                           = calc_pixsize2(file.RasterXSize,file.RasterYSize,gt)
    w = round(file.RasterXSize/(0.5/y_s))
    h = round(file.RasterYSize/(0.5/x_s))
    dest = path[:-4]+"_s.vrt"
    time.sleep(0.5)
    gdal.Warp(dest,path,width=w,format='VRT',height=h,resampleAlg='average',dstAlpha=True,dstNodata=255)  
    file_s                             = gdal.Open(dest)
    B_s                                = file_s.GetRasterBand(1).ReadAsArray()
    G_s                                = file_s.GetRasterBand(2).ReadAsArray()
    R_s                                = file_s.GetRasterBand(3).ReadAsArray()
    gt = file_s.GetGeoTransform()
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
    fsize                              = 14
    img_b                              = cv2.bilateralFilter(img_g,fsize,80,250)
    file_s = None
    gdal.Unlink(dest)
    pbar1.update(1)
    pbar1.close()
    return plist,img_s,img_b,mask_b,gt


"""
Takes the path to an orthomosaic and converts it to the path to the 
corresponding DEM. It rescales the file temporarily using GDAL, consequently 
ridges are extracted using filters of varying sizes.
---
plist    | list   | List for figures
path     | str    | Path to the orthomosaic up for georegistration
mask_b   | 2D arr | Binary map defining edges of orthomosaic                   (MaskB0F/MaskB1F)
gt       | tuple  | Geotransform corresponding to mask_b and ridges            (gt0F/gt1F)
ridges   | 2D arr | Binary map with edges or ridges found in DEM data          (Edges0F/Edges1F)  
"""
def OpenDEMs(plist,path):
    pbar1 = tqdm(total=1,position=0,desc="DemOpening")
    path_dem                           = path_to_path_dem(path)
    file                               = gdal.Open(path_dem)
    gt                                 = file.GetGeoTransform()
    x_s, y_s                           = calc_pixsize2(file.RasterXSize,file.RasterYSize,gt)
    w = round(file.RasterXSize/(0.05/y_s))
    h = round(file.RasterYSize/(0.05/x_s))
    dest = path_dem[:-4]+"_s.vrt"
    gdal.Warp(dest,path_dem,width=w,format='VRT',height=h,resampleAlg='average',dstAlpha=True,dstNodata=255)      
    file_s                             = gdal.Open(dest)   
    gt                                 = file_s.GetGeoTransform()
    dem                                = file_s.GetRasterBand(1).ReadAsArray() 
    mask                               = np.zeros(dem.shape)
    mask[dem==255]                     = 1
    dem_f = cv2.GaussianBlur(dem,(11,11),0)
    kernel = np.ones((15,15),np.float32)/(15**2)
    smooth = cv2.filter2D(dem,-1,kernel)
    ridges = (dem_f-smooth)
    mask_b = cv2.GaussianBlur(mask,(51,51),0)  
    ridges[mask_b>10**-10]=0  
    temp1 = np.zeros(ridges.shape)
    temp2 = np.zeros(ridges.shape)
    temp1[ridges<-0.01]=1
    temp2[ridges>-0.11]=1
    ridges = (temp1*temp2).astype(np.uint8) 
    p = plt.figure()
    plt.title('Ridges 0.05m')
    plt.imshow(ridges,cmap='Greys')
    file_s = None
    gdal.Unlink(dest)
    pbar1.update(1)
    pbar1.close()
    plt.close()
    plist.append(p)
    return plist,mask_b,gt,ridges


"""
Takes the current plist, places all the figures in a .pdf file, and erases the
plist. Also writes a small log on the receipt.txt file.
---
plist    | list   | List for figures
path     | str    | Path to the orthomosaic up for georegistration
base     | str    | Path to the orthomosaic that will be used as base
rec      | str    | Path to receipt
GCPstat  | str    | Contains the status of matches or GCP's after outlier removal
"""
def SaveFigs(plist,path,base,rec,GCPstat):
    pbar1 = tqdm(total=1,position=0,desc="CapFigures")
    dpiset = 1000
    filename = path.strip('.tif') + ('_LOG.pdf')
    if os.path.exists(filename.replace("\\","/")):
        os.remove(filename)   
    pp = PdfPages(filename)
    for fig in plist:
        fig.savefig(pp, format='pdf',dpi=dpiset)
    plist = np.array(plist)
    plist = []
    f = open(rec+"\\receipt.txt", "a")
    f.write("\n")
    f.write("\n"+time.strftime("%Y/%m/%d")+" - "+time.strftime("%H:%M"))
    f.write("\n"+"BASE: "+base)
    f.write("\n"+"FILE: "+path)
    f.write("\n"+GCPstat)
    f.close()
    pp.close()
    pbar1.update(1)
    pbar1.close()
    return plist


"""
When 60% or more of the grid has succesfully been matched the function moves 
the orthomosaic and its .vrt to the 'ready to upload' folder, the .points, DEM, 
and its .vrt are moved to the 'rectified dems and points' folder. Furthermore,
a copy of the DEM and its .vrt are also placed in the 'ready to upload' folder 
for possible use as base.
When the criteria is not met everything is moved to the 'not ready to upload' 
folder. The log is always moved to the 'receipt' folder.
---
path     | str    | Path to the orthomosaic up for georegistration
rtu      | str    | Path to ready to upload 
nrtu     | str    | Path to not ready to upload
dstr     | str    | Path to rectified dems and points
rec      | str    | Path to receipt
grid     | list   | List of 200 or 300 x,y tuples that form a grid in Edges0F
f2       | int    | Flag for creation of .vrt files
f3       | int    | Flag for moving files to corresponding folders
"""
def MoveFile(path,rtu,nrtu,dstr,rec,grid,f2):
    pbar1 = tqdm(total=1,position=0,desc="MoveFiles ")
    f3 = 0
    if f2 > 0.6*len(grid):
        shutil.move(path,rtu+"\\"+path_to_filename(path))
        shutil.move(path[:-4]+"-GR.vrt",rtu+"\\"+path_to_filename(path)[:-4]+"-GR.vrt")
        shutil.move(path_to_path_dem(path),dstr+"\\"+path_to_filename(path_to_path_dem(path)))
        shutil.move(path_to_path_dem(path)[:-4]+"-GR.vrt",dstr+"\\"+path_to_filename(path_to_path_dem(path))[:-4]+"-GR.vrt")   
        shutil.move(path[:-4]+".points",dstr+"\\"+path_to_filename(path)[:-4]+".points") 
        shutil.move(path[:-4]+"_LOG.pdf",rec+"\\"+path_to_filename(path)[:-4]+"_LOG.pdf") 
        f3 = 1                           
    else:
        shutil.move(path,nrtu+"\\"+path_to_filename(path))
        shutil.move(path_to_path_dem(path),nrtu+"\\"+path_to_filename(path_to_path_dem(path)))
        shutil.move(path[:-4]+"-GR.vrt",nrtu+"\\"+path_to_filename(path)[:-4]+"-GR.vrt")
        shutil.move(path_to_path_dem(path)[:-4]+"-GR.vrt",nrtu+"\\"+path_to_filename(path_to_path_dem(path))[:-4]+"-GR.vrt")
        shutil.move(path[:-4]+".points",nrtu+"\\"+path_to_filename(path)[:-4]+".points")
        shutil.move(path[:-4]+"_LOG.pdf",rec+"\\"+path_to_filename(path)[:-4]+"_LOG.pdf") 
        f3 = 2
    pbar1.update(1)
    pbar1.close()
    return f3


"""
###############################################################################
Meta functions below.
Used in METAA, CANNY, and RECCM, but not directly in Controller.py.
###############################################################################
"""


"""
Strips a full path to return only the name of the file.
"""
def path_to_filename(path):
    if "\\" in path:
        if "/" in path:
            index = min(path[::-1].find("\\"),path[::-1].find("/"))
            filename = path[::-1][:index][::-1]
        else:
            filename = path[::-1][:path[::-1].find("\\")][::-1]
    elif "/" in path:
        filename = path[::-1][:path[::-1].find("/")][::-1]
    return filename


"""
Retrieves the folder in which the file is placed.
"""
def path_to_folder(path):
    folpath,p = os.path.split(path)
    p,folder = os.path.split(folpath)
    return folder

"""
Extracts company, parcel, and date information from the filename.
"""
def filename_to_info(filename):
    company = " "
    parcel = " "
    date = " "
    file=filename[:-4]
    split = file.split("-")
    if len(split) == 3:
        company=split[0]
        parcel=split[1]
        date=split[2]
        if len(date) == 8:
            date = date+"0000"
    if len(split) == 4:
        if split[3] == "GR":
            company=split[0]
            parcel=split[1]
            date=split[2]
            if len(date) == 8:
                date = date+"0000"
    return company,parcel,date


"""
Adjusts the path of an orthomosaic to the corresponding path for the DEM.
"""
def path_to_path_dem(path):
    if path_to_folder(path) == "2_ready_to_upload":
        path = path.replace("2_ready_to_upload","00_rectified_DEMs_points")
    if path[-4:] == ".vrt":
        path_dem = path[:-4]
        if path_dem[-3:] == "-GR":
            path_dem = path_dem[:-3] + "_DEM-GR.vrt"
        else:
            path_dem = path_dem + "_DEM.vrt"
    elif path[-4:] == ".tif":
        path_dem = path[:-4]
        if path_dem[-3:] == "-GR":
            path_dem = path_dem[:-3] + "_DEM-GR.tif"
        else:
            path_dem = path_dem + "_DEM.tif"
    return path_dem


"""
Walks a folder top down and returns every orthomosaic that belongs to the same 
company and the same parcel, and also has a corresponding DEM.
"""
def walk_folder(folder,company,parcel):
    candidates = []
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            if "DEM" not in name:
                if name[-4:] == ".tif" or name[-4:] == ".vrt":
                    if filename_to_info(name)[0] == company:
                        if filename_to_info(name)[1] == parcel:
                            if os.path.exists(path_to_path_dem(os.path.join(root,name))) == True:
                                candidates.append(os.path.join(root,name).replace("\\","/"))
    return candidates


"""
Adjusted os.walk that allows for levels of top down.
"""
def walk_level(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


"""
Calculates the distance in meters for two pairs of lattitude and longitude.        
"""        
def calc_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 1000 * 6371 * c
    return m


"""
Calculates the size of a pixel in both the x and y direction based on the 
geotransform and array of a file.
"""
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


"""
Calculates the size of a pixel in both the x and y direction based on the 
geotransform and size of a file. (No need to load the file)
"""
def calc_pixsize2(s1,s2,gt):
    lon1 = gt[0] 
    lat1 = gt[3] 
    lon2 = gt[0] + gt[1]*s1
    lat2 = gt[3] + gt[4]*s1
    dist = calc_distance(lat1,lon1,lat2,lon2)
    ysize = dist/s1 
    lon2 = gt[0] + gt[2]*s2
    lat2 = gt[3] + gt[5]*s2
    dist = calc_distance(lat1,lon1,lat2,lon2)
    xsize = dist/s2
    return xsize, ysize 


"""
Returns the moving average of a function for a given window size.
"""
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


"""
Pixel linking function for CannyLines. Given an x,y it finds the next x,y based
on the orientation of pixels.
next1: Considers only pixels directly connected.
"""
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


"""
Pixel linking function for CannyLines. Given an x,y it finds the next x,y based
on the orientation of pixels.
next2: Considers only pixels in a directed cone based on the orientation of
        the line segment.
"""
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



"""
Pixel linking function for CannyLines. Given an x,y it finds the next x,y based
on the orientation of pixels.
next3: Same as next2, but the cone is bigger / more flexible.
"""
def next3(xSeed,ySeed,rows,cols,residualmap,boe,s,edgeChain):
    if s >= 3:
        if boe == 0:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]
        elif boe == 1:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]
    elif s < 3 and s > 1/3:
        if boe == 0:
            # 6
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1]#, 1, 0, 1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0,-1, 1]#, 0, 1, 1]
        elif boe == 1:
            # 2
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1]#,-1, 0,-1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0, 1,-1]#, 0,-1,-1]
    elif s >= -1/3 and s <= 1/3:
        if boe == 0:
            # 7
            X_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1, 1,-1, 0]
        elif boe == 1:
            # 3
            X_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]         
            Y_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2, 1,-1,-1, 1, 0]
    elif s < -1/3 and s > -3:
        if boe == 0:
            # 8
            X_OFFSET = [-1,-1, 0,-2,-1,-2, 0,-2, 1,-1]#, 1, 0, 1]
            Y_OFFSET = [ 1, 0, 1, 2, 2, 1, 2, 0, 1,-1]#, 0,-1,-1]
        elif boe == 1:
            # 4
            X_OFFSET = [ 1, 1, 0, 2, 1, 2, 0, 2,-1, 1]#,-1, 0,-1]
            Y_OFFSET = [-1, 0,-1,-2,-2,-1,-2, 0,-1, 1]#, 0, 1, 1]
    elif s <= -3:
        if boe == 0:
            # 1
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]         
            Y_OFFSET = [ 1, 1, 1, 2, 2, 2, 2, 2, 0, 0,-1,-1,-1]
        elif boe == 1:
            # 5
            X_OFFSET = [ 0,-1, 1, 0,-1, 1,-2, 2,-1, 1,-1, 1, 0]  
            Y_OFFSET = [-1,-1,-1,-2,-2,-2,-2,-2, 0, 0, 1, 1, 1]
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


"""
Pixel linking function for CannyLines. Given an x,y it finds the next x,y based
on the orientation of pixels.
next4: Assembles a cone based on the slope (s), which allows for even more 
        potential cones and a more accurate representation.
"""
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


"""
Uses the weighted least-squares estimator to fit the second 
order function f(x,y) = ax+by+cxy+dx^2+ey^2+f, equals offset.
"""
def hifit(origin_x,origin_y,CVa,offset):
    tmp_A = []
    tmp_b = []
    for i in range(len(origin_x)):
        tmp_A.append([origin_x[i], origin_y[i], origin_x[i]*origin_y[i], origin_x[i]**2, origin_y[i]**2, 1])
        tmp_b.append(offset[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    W = np.diag(CVa**-3)
    W = (W/np.sum(W))*len(CVa)
    fit = (A.T * W * A).I * A.T * W * b
    return fit


"""
###############################################################################
Archived functions below.
Mostly replaced by improved functions.
###############################################################################
"""


#def SelectFile(folder):
#    plt.close("all")
#    pathlist = []    
#    root = Tk()
#    root.withdraw()
#    #
#    root.filename =  filedialog.askopenfilename(multiple=True,initialdir = folder ,title = "Select Orthomosaics for Geo-Registration",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
#    #
#    for path in root.filename:
#        pathlist.append(path)
#    pathlist = sorted(pathlist, key = lambda a: filename_to_info(path_to_filename(a))[-1])
#    plist = []
#    plt.ioff()
#    return plist,pathlist
#
#def InboxxFiles(num):
#    plt.close("all")
#    metapath = []
#    for i in range(num):
#        path = []
#        root = Tk()
#        root.withdraw()
#        #
#        root.filename =  filedialog.askopenfilename(initialdir = r"D:\VanBovenDrive\VanBoven MT\Archive" ,title = "Select Base Orthomosaic",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
#        #
#        path.append(root.filename)
#        root.filename =  filedialog.askopenfilename(multiple=True, initialdir = r"C:\Users\VanBoven\Documents\100 Ortho Inbox" ,title = "Select Orthomosaics for Geo-Registration",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
#        for file in root.filename:
#            path.append(file)
#        metapath.append(path)
#    plist = []
#    plt.ioff()
#    return metapath,plist  
#
#def SelectFiles():
#    plt.close("all")
#    root = Tk()
#    root.withdraw()
#    #
#    root.filename =  filedialog.askopenfilename(initialdir = "D:\VanBovenDrive\VanBoven MT\Archive" ,title = "Select Base Orthomosaic",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
#    #
#    base = root.filename
#    path0 = base[base.find("Archive"):]
#    path1 = path0[path0.find("/")+1:]
#    path2 = path1[path1.find("/")+1:]
#    path3 = path2[path2.find("/")+1:]
#    folder = base[:base.find(path3)]
#    path = []
#    path.append(base)
#    date_base = re.findall(r"\d+",base)[-1]
#    for root, dirs, files in os.walk(folder, topdown=True):
#        for name in files:
#            if ".tif" in name:
#                if name not in base:
#                    if "_DEM" not in name:
#                        if float(date_base) < float(re.findall(r"\d+",name)[-1]):
#                            if "GR" in name:
#                                if os.path.exists(os.path.join(root,name).replace("-GR.tif","_DEM-GR.tif")) == True:
#                                    path.append(os.path.join(root,name).replace("\\","/"))
#                            else:    
#                                if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
#                                    path.append(os.path.join(root,name).replace("\\","/"))    
#    plist = []
#    plt.ioff()
#    return path,plist
#    
#def ChrInbFiles(num):
#    plt.close("all")
#    metapath = []
#    for i in range(num):
#        path = []
#        root = Tk()
#        root.withdraw()
#        #
#        root.filename =  filedialog.askopenfilename(initialdir = r"C:\Users\VanBoven\Documents\100 Ortho Inbox" ,title = "Select Orthomosaics for Geo-Registration",filetypes = (("GeoTiff files","*.tif"),("all files","*.*")))
#        #
#        inboxfile = root.filename
#        split = inboxfile.split("-")
#        company=split[0][::-1][:split[0][::-1].find("/")][::-1]
#        parcel=split[1]
#        date=split[2].replace(".tif","")        
#       
#        candidates = []
#        for root, dirs, files in os.walk(r"D:\VanBovenDrive\VanBoven MT\Archive", topdown=True):
#            for name in files:
#                if ".tif" in name:
#                    if "_DEM" not in name:
#                        if company in name:
#                            if parcel in name:
#                                if "GR" in name:
#                                    if os.path.exists(os.path.join(root,name).replace("-GR.tif","_DEM-GR.tif")) == True:
#                                        candidates.append(os.path.join(root,name).replace("\\","/"))
#                                else:    
#                                    if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
#                                        candidates.append(os.path.join(root,name).replace("\\","/"))    
#        dif = []
#        for file in candidates:
#            dif.append(float(date)-float(re.findall(r"\d+",file)[-1]))  
#        dif = np.array(dif)                                      
#        if (dif<=0).all() == True:
#            print("Found zero suitable orthomosaics in the past")
#        dif[dif<=0]=np.NaN
#        ind = np.where(dif==np.nanmin(dif))[0][0]
#        base = candidates[ind]
#        path.append(base)
#        path.append(inboxfile)
#        metapath.append(path)
#    plist = []
#    plt.ioff()
#    return metapath,plist
#    
#def OrtOpening(plist,path):
#    pbar1 = tqdm(total=1,position=0,desc="OrtOpening")
#    file                               = gdal.Open(path)
#    gt                                 = file.GetGeoTransform()
#    B                                  = file.GetRasterBand(1).ReadAsArray()
#    G                                  = file.GetRasterBand(2).ReadAsArray()
#    R                                  = file.GetRasterBand(3).ReadAsArray()
#    x_s, y_s                           = calc_pixsize(R,gt)
#    ps1 = 0.5
#    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
#    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
#    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/ps1)), int(B.shape[0]*(x_s/ps1))),interpolation = cv2.INTER_AREA)
#    fact_x_ps1                         = B.shape[0]/B_s.shape[0]
#    fact_y_ps1                         = B.shape[1]/B_s.shape[1]
#    img_s                              = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
#    mask                               = np.zeros(B_s.shape)
#    mask[R_s==255]                     = 1
#    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)  
#    img_s[:,:,0]                       = B_s
#    img_s[:,:,1]                       = G_s
#    img_s[:,:,2]                       = R_s
#    img_s_cielab                       = cv2.cvtColor(img_s, cv2.COLOR_BGR2Lab)
#    L                                  = img_s_cielab[:,:,0] 
#    hist                               = np.histogram(L[mask_b==0],bins=256)[0]
#    cdf                                = hist.cumsum()
#    cdf_m                              = np.ma.masked_equal(cdf,0)
#    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
#    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
#    L_eq                               = cdf[L]     
#    img_s_cielab_eq                    = img_s_cielab.copy()
#    img_s_cielab_eq[:,:,0]             = L_eq   
#    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
#    img_g                              = cv2.cvtColor(img_s_eq, cv2.COLOR_BGR2GRAY)
#    fsize                              = int(np.ceil((1.05/ps1))//2*2+1)
#    img_b                              = cv2.bilateralFilter(img_g,fsize,125,250)
#    pbar1.update(1)
#    pbar1.close()
#    return plist,img_s, img_b, mask_b, gt, fact_x_ps1, fact_y_ps1
#    
#def DemOpening(plist,path,Img0C):
#    pbar1 = tqdm(total=1,position=0,desc="DemOpening")
#    if "-GR.tif" in path:
#        temp = path.strip("-GR.tif")+"_DEM-GR.tif"
#    elif "_GR.vrt" in path:
#        temp = path.strip("_GR.vrt")+"_DEM_GR.vrt"
#    else:
#        temp = path.strip(".tif")+"_DEM.tif"
#    psF=0.05
#    file                               = gdal.Open(temp)
#    gt                                 = file.GetGeoTransform()
#    dem_o                              = file.GetRasterBand(1).ReadAsArray()
#    x_s, y_s                           = calc_pixsize(dem_o,gt)
#    mask                               = np.zeros(dem_o.shape)
#    if np.sum(dem_o==0) > np.sum(dem_o==np.min(dem_o)):
#        mask[dem_o == 0]               = 1
#    else:
#        mask[dem_o == np.min(dem_o)]   = 1
#    dem                                = cv2.resize(dem_o,(int(dem_o.shape[1]*(y_s/psF)), int(dem_o.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA)
#    mask                               = cv2.resize(mask,(int(mask.shape[1]*(y_s/psF)), int(mask.shape[0]*(x_s/psF))),interpolation = cv2.INTER_AREA) 
#    fx                                 = dem_o.shape[0]/dem.shape[0]
#    fy                                 = dem_o.shape[1]/dem.shape[1] 
#    dem_f = cv2.GaussianBlur(dem,(11,11),0)
#    smooth = cv2.GaussianBlur(dem_f,(15,15),0)
#    ridges = (dem_f-smooth)
#    #kernel = np.ones((n,n),np.float32)/(n**2)
#    #smooth = cv2.filter2D(dem_f,-1,kernel)
#    mask_b = cv2.GaussianBlur(mask,(51,51),0)  
#    ridges[mask>10**-10]=0  
#    temp1 = np.zeros(ridges.shape)
#    temp2 = np.zeros(ridges.shape)
#    temp1[ridges<-0.01]=1
#    temp2[ridges>-0.11]=1
#    ridges = (temp1*temp2).astype(np.uint8) 
#    p = plt.figure()
#    plt.title('Ridges 0.05m')
#    plt.imshow(ridges,cmap='Greys')
#    pbar1.update(1)
#    pbar1.close()
#    plt.close()
#    plist.append(p)
#    return plist,gt,fx,fy,mask_b,ridges
#
#def rangemaker(num,thMeaningfulLength):
#    span = int(thMeaningfulLength/5)
#    range_array = np.zeros(2*span+1)
#    for i in range(len(range_array)):
#        range_array[i]=int(num-span+i)
#    return range_array
#    
#def fit(origin_x,origin_y,CVa,offset):
#    tmp_A = []
#    tmp_b = []
#    for i in range(len(origin_x)):
#        tmp_A.append([origin_x[i], origin_y[i], 1])
#        tmp_b.append(offset[i])
#    b = np.matrix(tmp_b).T
#    A = np.matrix(tmp_A)
#    W = np.diag(CVa**-3)
#    W = (W/np.sum(W))*len(CVa)
#    fit = (A.T * W * A).I * A.T * W * b
#    return fit