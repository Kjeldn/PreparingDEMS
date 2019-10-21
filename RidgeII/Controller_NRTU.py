import METAA

import os
import shutil

folder  = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\4_not_ready_to_upload"
rtu     = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\2_ready_to_upload"
dstr    = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\00_rectified_DEMs_points"
rec     = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\9_receipt"

pathlist = []
for root, dirs, files in os.walk(folder, topdown=True):
    for name in files:
        if "tif" in name:
            if "GR" in name:
                if os.path.exists(os.path.join(root,name).replace("-GR.tif","_DEM-GR.tif")) == True:
                    if os.path.exists(os.path.join(root,name).replace(".tif",".points")) == True:
                        pathlist.append(os.path.join(root,name).replace("\\","/"))    
            else:
                if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
                    if os.path.exists(os.path.join(root,name).replace(".tif",".points")) == True:
                        pathlist.append(os.path.join(root,name).replace("\\","/"))  
del root,dirs,files,name
    
for path in pathlist:
    shutil.move(path,rtu+"\\"+METAA.path_to_filename(path))
    shutil.move(path[:-4]+"-GR.vrt",rtu+"\\"+METAA.path_to_filename(path)[:-4]+"-GR.vrt")
    shutil.move(METAA.path_to_path_dem(path),dstr+"\\"+METAA.path_to_filename(METAA.path_to_path_dem(path)))
    shutil.move(METAA.path_to_path_dem(path)[:-4]+"-GR.vrt",dstr+"\\"+METAA.path_to_filename(METAA.path_to_path_dem(path))[:-4]+"-GR.vrt")   
    shutil.move(path[:-4]+".points",dstr+"\\"+METAA.path_to_filename(path)[:-4]+".points") 
    shutil.move(path[:-4]+"_LOG.pdf",rec+"\\"+METAA.path_to_filename(path)[:-4]+"_LOG.pdf") 