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
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt

wd = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path = "c01_verdonk-Wever oost-201908041528_DEM-GR"
path_ahn = None #"m_19fn2.tif"
plants = []

#%% plants

with fiona.open(wd + "/20190717_count.shp") as src:
    plants = np.zeros((2, len(src)))
    for i in range(0, len(src), 1000):
        plants[:, i] = [src[i]['geometry']['coordinates'][0][0], src[i]['geometry']['coordinates'][0][1]] if src[i]['geometry'] else [0,0]
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema
        
for i in range(plants.shape[1]-1, -1, -1):
    if plants[0][i]==0 and plants[1][i] == 0:
        plants = np.delete(plants, i, 1)
        
vor = Voronoi(plants)
voronoi_plot_2d(vor)
plt.show()
