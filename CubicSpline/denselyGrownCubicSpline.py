import gdal
import fiona
import util
import numpy as np
from scipy import interpolate
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

wd = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path = "c01_verdonk-Wever oost-201908041528_DEM-GR"
path_ahn = None #"m_19fn2.tif"
plants = []

if path_ahn:
    orig = gdal.Open(wd + "/" + path + ".tif")
    dst_shape = (orig.GetRasterBand(1).YSize, orig.GetRasterBand(1).XSize)
    dst_transform = A(orig.GetGeoTransform()[1], 0, orig.GetGeoTransform()[0], 0, orig.GetGeoTransform()[5],  orig.GetGeoTransform()[3])
    ahn_array = np.zeros(dst_shape)
    dst_crs = "EPSG:4326"
    
    with rasterio.open(wd + "/" + path_ahn) as src:
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

with fiona.open(wd + "/20190717_count.shp") as src:
    for s in src:
        plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

src_schema['properties']['diff'] = 'float:24.15'

x = []
y = []
values = []

tif = gdal.Open(wd + "/" + path + ".tif")
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
        if mask[i][j] == 0 and polygon.contains(Point(i, j)):
            e.append(a[i][j] if path_ahn==None else a[i][j] - ahn_array[i][j])
mask = None

util.create_tiff(a - np.mean(e), gt, proj, wd + "\\" + path + "cubic_dense.tif")