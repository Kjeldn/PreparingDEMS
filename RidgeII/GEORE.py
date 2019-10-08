import os
import csv
import gdal
from tqdm import tqdm
from io import StringIO
from shutil import move
from tempfile import mkstemp

#folder  = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\1_ready_to_rectify'
#folder  = r'D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris'
#folder  = r'D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer'
#folder  = r'D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter'

def Georegistr(file):
    pbar1 = tqdm(total=1,position=0,desc="CreateVRT ")
    flag = 0
    if "\\" in file:
        folder = file[::-1][file[::-1].find("\\"):][::-1]
    else:
        folder = file[::-1][file[::-1].find("/"):][::-1]
    path = []
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            if "tif" in name:
                if "GR" in name:
                    if os.path.exists(os.path.join(root,name).replace("-GR.tif","_DEM-GR.tif")) == True:
                        if os.path.exists(os.path.join(root,name).replace(".tif",".points")) == True:
                            path.append(os.path.join(root,name).replace("\\","/"))    
                else:
                    if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
                        if os.path.exists(os.path.join(root,name).replace(".tif",".points")) == True:
                            path.append(os.path.join(root,name).replace("\\","/"))     
    del root,dirs,files,name
    
    pattern = "    <SourceDataset relativeToVRT=\"0\"></SourceDataset>"
    
    for tif in path:
        if "GR" in tif:
            dem = tif.replace("-GR.tif","_DEM-GR.tif")
        else:
            dem = tif.replace(".tif","_DEM.tif")
        points = tif.replace(".tif",".points")
        
        gt_t = gdal.Open(tif).GetGeoTransform()
        gt_d = gdal.Open(dem).GetGeoTransform()
        gcp_t = []
        gcp_d = []
        
        points = open(points,"r")
        points = points.readlines()[1:]
        for line in points:
            l = StringIO(line)
            reader = csv.reader(l, delimiter=',')
            for row in reader:
                gcp_t.append(gdal.GCP(float(row[0]), float(row[1]), 0, (float(row[2])-gt_t[0])/gt_t[1], (float(row[3])-gt_t[3])/gt_t[5]))
                gcp_d.append(gdal.GCP(float(row[0]), float(row[1]), 0, (float(row[2])-gt_d[0])/gt_d[1], (float(row[3])-gt_d[3])/gt_d[5]))
        del points,gt_t,gt_d,line,row,reader
        
        wops = gdal.WarpOptions(format='VRT',
                             dstAlpha=True,   
                             srcSRS = 'EPSG:4326',
                             dstSRS = 'EPSG:4326',
                             warpOptions=['NUM_THREADS=ALL_CPUS'],
                             warpMemoryLimit=3000,
                             creationOptions=['COMPRESS=LZW','TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS', 'JPEG_QUALITY=100', 'BIGTIFF=YES', 'ALPHA=YES'],
                             resampleAlg='cubicspline',
                             multithread=True,
                             tps=True,
                             transformerOptions=['NUM_THREADS=ALL_CPUS'])
        
        tops = gdal.TranslateOptions(format='VRT',outputType=gdal.GDT_Byte,outputSRS='EPSG:4326',GCPs=gcp_t)
        tif_d = tif.replace(".tif","-GR.vrt")
        subst   = "    <SourceDataset relativeToVRT=\"1\">"+tif[::-1][:tif[::-1].find("/")][::-1]+"</SourceDataset>"
        temp = gdal.Translate('',tif,options=tops)
        gdal.Warp(tif_d,temp,options=wops)
        fh, abs_path = mkstemp()
        with os.fdopen(fh,'w') as new_file:
            with open(tif_d) as old_file:
                for line in old_file:
                    new_file.write(line.replace(pattern, subst))
        os.remove(tif_d)
        move(abs_path, tif_d)
        
        tops = gdal.TranslateOptions(format='VRT',outputType=gdal.GDT_Float32,outputSRS='EPSG:4326',GCPs=gcp_d)
        dem_d = dem.replace(".tif","-GR.vrt")
        subst   = "    <SourceDataset relativeToVRT=\"1\">"+dem[::-1][:dem[::-1].find("/")][::-1]+"</SourceDataset>"
        temp = gdal.Translate('',dem,options=tops)
        gdal.Warp(dem_d,temp,options=wops)
        fh, abs_path = mkstemp()
        with os.fdopen(fh,'w') as new_file:
            with open(dem_d) as old_file:
                for line in old_file:
                    new_file.write(line.replace(pattern, subst))
        os.remove(dem_d)
        move(abs_path, dem_d)
        flag = 1
        pbar1.update(1)
        pbar1.close()
        return flag
