import gdal
import util
import numpy as np
import fiona
import matplotlib.pyplot as plt
import divide_into_beds as dib

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
    
# =============================================================================
# plt.plot(np.arange(len(planes)), [np.median(heights[:,i]) for i in range(len(planes))])
# plt.fill_between(np.arange(len(plants)), [np.percentile(heights[:,i], 25) for i in range(len(planes))], [np.percentile(heights[:,i], 75) for i in range(len(planes))])
# plt.show()
# =============================================================================

beds = dib.divide(plants)
heights_beds = []
for bed in beds:
    heights_array = np.array((len(bed), len(paths)))
    for i in range(len(bed)):
        heights[i,:] = np.array([plane.getMaxValueAt(bed[i][1], bed[i][0]) for plane in planes])
    heights_beds.append(heights_array)
    
for heights_bed in heights_beds:
    plt.plot(np.arange(len(planes)), [np.median(heights_bed[:,i]) for i in range(len(planes))])
plt.show()