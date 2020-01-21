import numpy as np
import cupy as cp
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon,Point
import geopandas
import gdal
import matplotlib.pyplot as plt
from tqdm import tqdm

mask_path = r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\archive\c01_verdonk-Wever west-201907170749-GR_clustering_output.tif"
plant_path = r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Wever west\20190717\0749\Plant_count\20190717_count.shp"

plants = geopandas.read_file(plant_path)
plants = plants.mask(plants.geometry.eq('None')).dropna()

plants_r = np.array([(p.centroid.xy[0][0], p.centroid.xy[1][0]) for p in plants.loc[:,'geometry']])
vor = Voronoi(plants_r)

file = gdal.Open(mask_path)
gt   = file.GetGeoTransform()
band = file.GetRasterBand(1)
mask = band.ReadAsArray()
vora = np.zeros(mask.shape)

size_x = np.max(mask[:,0].shape)
size_y = np.max(mask[0,:].shape)

num = 0
pbar = tqdm(total=len(vor.regions), desc="voronoi array", position=0)
for region in vor.regions:
    pbar.update(1)
    if len(region) > 0 and all([i > 0 for i in region]):
        num += 1
        corners_WGS84 = np.array([vor.vertices[i] for i in region])
        corners_index = np.zeros(corners_WGS84.shape)
        corners_index[:,0] = (corners_WGS84[:,0]-gt[0])/gt[1]
        corners_index[:,1] = (corners_WGS84[:,1]-gt[3])/gt[5]
        x_min = int(np.floor(np.min(corners_index[:,0])))
        x_max = int(np.ceil(np.max(corners_index[:,1])))
        y_min = int(np.floor(np.min(corners_index[:,0])))
        y_max = int(np.ceil(np.max(corners_index[:,1])))
        for x in range(np.max(0,x_min),np.min(size_x,x_max)):
            for y in range(np.max(0,y_min),np.min(size_y,y_max)):
                vora[x,y]=num

pbar = tqdm(total=1, desc="numpy", position=0)
shapes = np.multiply(mask,vora)
pbar.update(1)
pbar.close()
pbar = tqdm(total=1, desc="cupy", position=0)
shapes = cp.multiply(mask,vora)
pbar.update(1)
pbar.close()
