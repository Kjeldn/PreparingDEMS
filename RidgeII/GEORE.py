import os
import csv
import gdal
from tqdm import tqdm
from io import StringIO
from shutil import move
from tempfile import mkstemp

folder = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\1_ready_to_rectify'
path = []
for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            if ".tif" in name:
                    if "_DEM" not in name:
                        if os.path.exists(os.path.join(root,name).replace(".tif","_DEM.tif")) == True:
                            if os.path.exists(os.path.join(root,name).replace(".tif",".points")) == True:
                                path.append(os.path.join(root,name).replace("\\","/"))     
del root,dirs,files,name

for file in path:
    dem = file.replace(".tif","_DEM.tif")
    points = file.replace(".tif",".points")
    gt_t = gdal.Open(file).GetGeoTransform()
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
    
    file_d = 
    dem_d = 
    
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