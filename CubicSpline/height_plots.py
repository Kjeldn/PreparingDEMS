import gdal
import util
import numpy as np
import fiona
import matplotlib.pyplot as plt

wd = ""
paths = ["c01_verdonk-Rijweg stalling 1-201907091137_DEM-GR_cubic", 
         "c01_verdonk-Rijweg stalling 1-201907170849_DEM-GR_cubic",
         "c01_verdonk-Rijweg stalling 1-201907230859_DEM-GR_cubic",
         "c01_verdonk-Rijweg stalling 1-201908051539_DEM-GR_cubic"]
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
                planes.append(s['geometry']['coordinates'])
            elif s['geometry']['type'] == 'MultiPoint':
                planes.append(s['geometry']['coordinates'][0])

heights = np.array((len(plants), len(paths)))
for i in range(len(plants)):
    heights[i,:] = np.array([plane.getMaxValueAt(plants[i][1], plants[i][0]) for plane in planes])
    
plt.plot(np.arange(len(plants)), [np.mean(heights[:,i]) for i in range(len(plants))])
plt.fill_between(np.arange(len(plants)), [np.mean(heights[:,i]) - 2*np.std(heights[:,i]) for i in range(len(plants))], [np.mean(heights[:,i]) + 2*np.std(heights[:,i]) for i in range(len(plants))])