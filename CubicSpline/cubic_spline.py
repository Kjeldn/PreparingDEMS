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
from tkinter import filedialog
from tkinter import *

gdal.UseExceptions()

root = Tk()
paths = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select dems", parent=root, multiple=True)
plant_path = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select plant count", parent=root)

use_ahn = False
if use_ahn:
    ahn_path = filedialog.askopenfilename(initialdir =  r"D:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select ahn dem", parent=root)

root.destroy()
use_ridges = True
plot = False

##The space between possible bare ground points to fit over
step = 40

#%% plants
plants = []
    
with fiona.open(plant_path) as src:
    for s in src:
        if s['geometry']:
            if s['geometry']['type'] == 'Point':
                plants.append(s['geometry']['coordinates'] if s['geometry'] else None)
            if s['geometry']['type'] == 'MultiPoint':
                plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema
    
#%% reproject AHN Model
if use_ahn:
    orig = gdal.Open(paths[0])
    dst_shape = (orig.GetRasterBand(1).YSize, orig.GetRasterBand(1).XSize)
    dst_transform = A(orig.GetGeoTransform()[1], 0, orig.GetGeoTransform()[0], 0, orig.GetGeoTransform()[5],  orig.GetGeoTransform()[3])
    ahn_array = np.zeros(dst_shape)
    dst_crs = "EPSG:4326"
    
    with rasterio.open(ahn_path) as src:
        source = src.read(1)
        
        with rasterio.Env():
            reproject(
                    source,
                    ahn_array,
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = dst_transform,
                    dst_crs = dst_crs,
                    respampling = Resampling.nearest
                    )
            
    source = None
#%% cubic spline
pbar = tqdm(total=len(paths), desc="Doing cubic spline thingies", position=0)
for a in range(len(paths)):
    file = gdal.Open(paths[a])
    
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
    poly = Polygon(zip(x_plants, y_plants))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line)
        
    if use_ridges:
        ridges_array = dt.get_ridges_array(array, -0.01).astype(np.uint8)
    
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
            data[i][j] = array[step * i, step * j] - ahn_array[step * i, step * j] if use_ahn else array[step * i, step * j]
            x[i][j] = step * i
            y[i][j] = step * j
            if array[step * i, step * j] == 0 or not polygon.contains(Point(step * i, step * j)):
                mask[i][j] = True
            if use_ridges and ridges_array[step * i, step * j] != 1:
                mask[i][j] = True
            if use_ahn and abs(ahn_array[step * i, step * j]) > 10:
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