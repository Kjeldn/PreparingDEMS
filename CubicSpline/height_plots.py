import gdal
import util
import numpy as np
import fiona
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import divide_into_beds as dib

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
plt.scatter(np.array(plants)[:,0], np.array(plants)[:,1], c=z, camp="reds")
plt.show()

plt.hist(heights[:,0],bins=1000)
plt.hist(heights[:,1],bins=1000)
plt.hist(heights[:,2],bins=1000)
plt.show()

