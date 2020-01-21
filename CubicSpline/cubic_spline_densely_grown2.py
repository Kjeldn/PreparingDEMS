"""
This script corrects DEMs by removing the trend. It should not be used when the crop field is densely grown.
Because the error in the DEM is determined by assessing the error in GCP's. These GCP's are located between crop rows on
the bare ground and are not visible if the field is densely grown.
The destination path of the corrected dem is original_path.split(".")[0] + "_cubic.tif"
"""

import gdal
import numpy as np
from scipy import interpolate
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import Affine as A
import detect_ridges as dt
import util_cubic as util
import fiona
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt
from tqdm import tqdm
gdal.UseExceptions()

paths = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190710\1007\Orthomosaic\c07_hollandbean-Joke Visser-201907101007_DEM-GR.vrt",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190802\0829\Orthomosaic\c07_hollandbean-Joke Visser-201908020829_DEM-GR.vrt",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190823\1004\Orthomosaic\c07_hollandbean-Joke Visser-201908231004_DEM-GR.vrt",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190830\0729\Orthomosaic\c07_hollandbean-Joke Visser-201908300729_DEM-GR.vrt",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190906\0802\Orthomosaic\c07_hollandbean-Joke Visser-201909060802_DEM-GR.vrt",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190701\0933\Orthomosaic\c07_hollandbean-Joke Visser-201907010933_DEM-GR.vrt"]
plant_path = r"D:\800 Operational\c07_hollandbean\Joke visser\Count\906250739-GR_plant_count.gpkg"
base_path = r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190619\1208\Orthomosaic\c07_hollandbean-Joke Visser-201906191208_DEM-GR.vrt"

use_ridges = True
plot = False

##The space between possible bare ground points to fit over
step = 20

#%% plants
plants = []
    
with fiona.open(plant_path) as src:
    for s in src:
        if s['geometry']:
            if s['geometry']['type'] == 'Point':
                plants.append(s['geometry']['coordinates'] if s['geometry'] else None)
            if s['geometry']['type'] == 'MultiPoint':
                plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)

#%% cubic spline
pbar = tqdm(total=len(paths), desc="Doing cubic spline thingies", position=0)
for a in range(len(paths)):
    file = gdal.Open(paths[a])
    dst_shape = (file.GetRasterBand(1).YSize, file.GetRasterBand(1).XSize)
    dst_transform = A(file.GetGeoTransform()[1], 0, file.GetGeoTransform()[0], 0, file.GetGeoTransform()[5],  file.GetGeoTransform()[3])
    base_array = np.zeros(dst_shape)
    dst_crs = "EPSG:4326"
    
    with rasterio.open(base_path) as src:
        source = src.read(1)
        
        with rasterio.Env():
            reproject(
                    source,
                    base_array,
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = dst_transform,
                    dst_crs = dst_crs,
                    respampling = Resampling.cubic
                    )
            
    source = None
    
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    projection = file.GetProjection()
    gt = file.GetGeoTransform()
    xsize = band.XSize
    ysize = band.YSize
    
    x_plants = []
    y_plants = []
    values = []

    plane = util.Plane(array, gt)
    for p in plants:
        if p:
            xc_plant, yc_plant = plane.getIndicesByCoord(p[1], p[0])
            x_plants.append(xc_plant)
            y_plants.append(yc_plant)
    convex_hull = Polygon(zip(x_plants, y_plants)).convex_hull
        
    if use_ridges:
        ridges_array = dt.get_ridges_array(array, -0.035).astype(np.uint8)
        mask = util.getMask(array, plants, gt, k_size = 13)
        ridges_array *= mask
    
    ##Remove all non-values from array
    array[array == np.amin(array)] = 0
    
    data = np.zeros((int(ysize/step), int(xsize/step)))
    mask = np.zeros((int(ysize/step), int(xsize/step))) > 0
    x = np.zeros((int(ysize/step), int(xsize/step)))
    y = np.zeros((int(ysize/step), int(xsize/step)))
    
    xx = []
    yy = []
    
    # create list of points inside the field to get the fit over
    for i in range(int(ysize/step)):
        for j in range(int(xsize/step)):
            data[i][j] = array[step * i, step * j] - base_array[step * i, step * j]
            x[i][j] = step * i
            y[i][j] = step * j
            if array[step * i, step * j] == 0 or not convex_hull.contains(Point(step * i, step * j)):
                mask[i][j] = True
            if use_ridges and ridges_array[step * i, step * j] != 1:
                mask[i][j] = True
            if abs(base_array[step * i, step * j]) > 10:
                mask[i][j] = True
            if not mask[i][j]:
                xx.append(step*i)
                yy.append(step*j)
                  
    if plot:
        plt.figure()
        plt.scatter(yy, xx)
        plt.spy(ridges_array)
        plt.show()
    ridges_array = None
    
    z = np.ma.array(data, mask=mask)
    data = None
    mask = None
    
    ##Remove all points which are either a non-value, not bare ground, non-value in AHN or not in the polygon
    z1 = z[~z.mask]
    y1 = y[~z.mask]
    x1 = x[~z.mask]

    z = None
    y = None
    x = None
    
    xnew, ynew = np.mgrid[0:ysize, 0:xsize]
    tck, fp, ier, msg = interpolate.bisplrep(x1, y1, z1, full_output = 1)
    znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    xnew = None
    ynew = None
    z1 = None
    y1 = None
    x1 = None
    
    znew = array - znew
    array = None
    
    util.create_tiff(znew, gt, projection, paths[a].split(".")[0] + "_cubic.tif")
    znew = None
    pbar.update(1)
pbar.close()