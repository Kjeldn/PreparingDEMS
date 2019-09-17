import gdal
import util
import numpy as np
import fiona
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import divide_into_beds as dib
from shapely.geometry import Polygon, Point
import colorsys

wd = r"D:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants compare"
paths = ["c01_verdonk-Rijweg stalling 1-201907091137_DEM-GR_cubic", 
         "c01_verdonk-Rijweg stalling 1-201907170849_DEM-GR_cubic",
         "c01_verdonk-Rijweg stalling 1-201907230859_DEM-GR_cubic",
         "c01_verdonk-Rijweg stalling 1-201908051539_DEM-GRcubic"]
plants_count_path = "20190709_count"
planes = []

for path in paths:
    file = gdal.Open(wd + "/" + path + ".tif")
    gt = file.GetGeoTransform()
    projection = file.GetProjection()
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    planes.append(util.Plane(array, gt))
    file = None

plants = []
with fiona.open(wd + "/" + plants_count_path + ".shp") as src:
    for s in src:
        if s['geometry']:
            if s['geometry']['type'] == 'Point':
                plants.append(s['geometry']['coordinates'])
            elif s['geometry']['type'] == 'MultiPoint':
                plants.append(s['geometry']['coordinates'][0])

heights = np.zeros((len(plants), len(paths)))
for i in range(len(plants)):
    heights[i,:] = np.array([plane.getMaxValueAt(plants[i][1], plants[i][0], k_size=15) for plane in planes])
    
plt.plot(np.arange(len(planes)), [np.median(heights[:,i]) for i in range(len(planes))])
plt.fill_between(np.arange(len(planes)), [np.percentile(heights[:,i], 25) for i in range(len(planes))], [np.percentile(heights[:,i], 75) for i in range(len(planes))], color="cyan")
plt.show()
# =============================================================================
# 
# beds,_ = dib.divide(np.array(plants))
# heights_beds = []
# for bed in beds:
#     heights_array = np.zeros((len(bed), len(paths)))
#     for i in range(len(bed)):
#         heights_array[i,:] = np.array([plane.getMaxValueAt(bed[i][1], bed[i][0]) for plane in planes])
#     heights_beds.append(heights_array)
#     
# for heights_bed in heights_beds:
#     plt.plot(np.arange(len(planes)), [np.median(heights_bed[:,i]) for i in range(len(planes))])
# plt.show()
# =============================================================================

z = heights[:,1] - heights[:,0]
plt.scatter(np.array(plants)[:,0], np.array(plants)[:,1], c=z, cmap="Reds")
plt.show()

plt.hist(heights[:,0],bins=1000)
plt.hist(heights[:,1],bins=1000)
plt.hist(heights[:,2],bins=1000)
plt.show()

beds = []

with fiona.open(wd + "/c01_verdonk-Rijweg stalling 1-201907170849-GR.shp") as src:
    for s in src:
        beds.append(Polygon(s['geometry']['coordinates'][0]))
        
height_beds = [[] for i in range(len(beds))]
for i, plant in enumerate(plants):
    for j, bed in enumerate(beds):
        if Point(plant).within(bed):
            height_beds[j].append(heights[i,1] - heights[i,0])
            
max_mean = 0
min_mean = 100
for bed in height_beds:
    if np.mean(bed) > max_mean:
        max_mean = np.mean(bed)
    if np.mean(bed) < min_mean:
        min_mean = np.mean(bed)
    
fig = plt.figure()
for i, bed in enumerate(beds):
    plt.fill(*bed.exterior.xy, c=(np.mean(height_beds[i])/max_mean,0,0))
fig.show()


