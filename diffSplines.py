import util
import gdal
import numpy as np
import fiona
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt

wd = r"C:/Users/wytze/OneDrive/Documents/vanBoven/Broccoli"
paths = ["c01_verdonk-Wever oost-201907240707_DEM-GR_cubic", "c01_verdonk-Wever oost-201907170731_DEM-GR_cubic", "cubic_dense"]

plants = []

with fiona.open(wd + "/20190717_count.shp") as src:
    for s in src:
        plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema
    
planes = []
for p in paths:
    file = gdal.Open(wd + "/" + p + ".tif")
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    xsize = band.XSize
    ysize = band.YSize
    gt = file.GetGeoTransform()
    proj = file.GetProjection()
    
    planes.append(util.Plane(array, gt))
    
h = []

    
mask = util.getMask(np.zeros(planes[0].array.shape), plants, gt)
x = []
y = []

for i in range(0, len(plants), 20):
    if plants[i]:
        xx, yy = planes[1].getIndicesByCoord(plants[i][1], plants[i][0])
        x.append(xx)
        y.append(yy)
    
poly = Polygon(zip(x, y))
poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
polygon = Polygon(poly_line.coords)

for i in range(0, ysize, 30):
    for j in range(0, xsize, 30):
        if mask[i][j] == 0 and polygon.contains(Point(i, j)):
            xx, yy = planes[1].getCoordByIndices(i,j)
            h.append(planes[1].array[i][j] - planes[2].getMaxValueAt(xx, yy))
            
mask = None
array = None

plt.boxplot(h, sym='')
plt.show()