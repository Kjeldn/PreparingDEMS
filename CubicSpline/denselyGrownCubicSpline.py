import gdal
import fiona
import util_cubic as util
import numpy as np
from scipy import interpolate
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import detect_ridges as dr
from tkinter import filedialog
from tkinter import *
from tqdm import trange

<<<<<<< Updated upstream
use_ahn = False
=======
use_ahn = False #"m_19fn2.tif"
>>>>>>> Stashed changes

root = Tk()
paths = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select dems", parent=root, multiple=True)
plant_path = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select plant count", parent=root)

if use_ahn:
    ahn_path = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select ahn dem", parent=root)

root.destroy()
plants = []

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

src_schema['properties']['diff'] = 'float:24.15'

for a in trange(len(paths), desc="cubic spline thingies"):
    path = paths[a]
    x = []
    y = []
    values = []
    
    tif = gdal.Open(path)
    band = tif.GetRasterBand(1)
    array = band.ReadAsArray()
    array[array == np.amin(array)] = 0
    xsize = band.XSize
    ysize = band.YSize
    gt = tif.GetGeoTransform()
    proj = tif.GetProjection()
    plane = util.Plane(array, gt)
    tif = None
    
    for i in range(0, len(plants), 20):
        if plants[i]:
            values.append(plane.getMaxValueAt(plants[i][1], plants[i][0]))
            xx, yy = plane.getIndicesByCoord(plants[i][1], plants[i][0])
            x.append(xx)
            y.append(yy)    
            
    poly = Polygon(zip(x, y))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    
    xnew, ynew = np.mgrid[0:ysize, 0:xsize]
    
    tck = interpolate.bisplrep(x, y, values)
    znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    ridges_array = dr.get_ridges_array(array)
    x = None
    y = None
    xnew = None
    ynew = None
    values = None
    a = (array - znew)
    array = None
    znew = None
    
    mask = util.getMask(a, plants, gt)
    
    
    e = []
    for i in range(0, a.shape[0], 20):
        for j in range(0, a.shape[1], 20):
            if mask[i][j] == 0 and ridges_array[i][j] == 1 and polygon.contains(Point(i, j)):
                e.append(a[i][j] if not use_ahn else a[i][j] - ahn_array[i][j])
<<<<<<< Updated upstream
    mask = None
=======
    #mask = None
    
    util.create_tiff(a - np.mean(e)-0.1, gt, proj, path.split(".")[0] + "_cubic.tif")
>>>>>>> Stashed changes
    
#%%
file = gdal.Open(r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare\Aart Maris\c07_hollandbean-Aart Maris-201907241028_DEM_GR_cubic.tif")
band = file.GetRasterBand(1)
array = band.ReadAsArray()
gt = file.GetGeoTransform()
proj = file.GetProjection()
array -= 1
util.create_tiff(array, gt, proj, r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare\Aart Maris\c07_hollandbean-Aart Maris-201907241028_DEM_GR_cubic.tif")