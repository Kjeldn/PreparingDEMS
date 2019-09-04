import gdal
import fiona
import util
import numpy as np
import matplotlib.pyplot as plt

class Plane:
    def __init__(self, array, gt):
        self.array = array
        self.gt = gt
        
    def getMeanValueAt(self, x, y, k_size = 3):
        ptl = util.Point(int(abs(np.floor((x - self.gt[3])/self.gt[5]))), int(abs(np.floor((y - self.gt[0])/self.gt[1]))))
        sum_k = 0
        for i in range(ptl.x - int((k_size - 1)/2), ptl.x + int((k_size - 1)/2) + 1):
            for j in range(ptl.y - int((k_size - 1)/2), ptl.y + int((k_size - 1)/2) + 1):
                sum_k += self.array[i][j]
        return sum_k/(k_size**2)
    
    def getMaxValueAt(self, x, y, k_size = 7):
        ptl = util.Point(int(abs(np.floor((x - self.gt[3])/self.gt[5]))), int(abs(np.floor((y - self.gt[0])/self.gt[1]))))
        return np.amax(self.array[ptl.x - int((k_size - 1)/2):ptl.x + int((k_size - 1)/2) + 1, ptl.y - int((k_size - 1)/2):ptl.y + int((k_size - 1)/2) + 1])

#%% Diff between first two
wr = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path = ["c01_verdonk-Wever oost-201907170731_DEM-GR_cubic.tif", 
        "c01_verdonk-Wever oost-201907240707_DEM-GR_cubic.tif"]
plants = []

with fiona.open(wr + "/20190717_count.shp") as src:
    for s in src:
        plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

src_schema['properties']['diff'] = 'float:24.15'

values = np.zeros((len(path), len(plants)))
planes = []

for p in path:
    tif = gdal.Open(wr + "/" + p)
    planes.append(Plane(tif.GetRasterBand(1).ReadAsArray(), tif.GetGeoTransform()))
    
for i in range(len(plants)):
    for j in range(len(planes)):
        if plants[i]:
            values[j, i] = planes[j].getMaxValueAt(plants[i][1], plants[i][0])

plt.boxplot(values[1,:] - values[0,:], sym='')
plt.show()

with fiona.open(wr + "/20190717_count.shp") as src:

    with fiona.open(wr + "/20190717_count_diff.shp", 'w', driver=src_driver, crs=src_crs, schema=src_schema) as dest:
        i = 0
        for d in src:
            d['properties']['diff'] = values[1, i] - values[0, i]
            i += 1
            dest.write(d)

#%% Diff between last two
wr = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path = ["c01_verdonk-Wever oost-201907240707_DEM-GR_cubic.tif",
        "cubic_dense.tif"]
plants = []

with fiona.open(wr + "/20190717_count.shp") as src:
    for s in src:
        plants.append(s['geometry']['coordinates'][0] if s['geometry'] else None)
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema

src_schema['properties']['diff'] = 'float:24.15'

values = np.zeros((len(path), len(plants)))
planes = []

for p in path:
    tif = gdal.Open(wr + "/" + p)
    planes.append(Plane(tif.GetRasterBand(1).ReadAsArray(), tif.GetGeoTransform()))
    
for i in range(len(plants)):
    for j in range(len(planes)):
        if plants[i]:
            values[j, i] = planes[j].getMaxValueAt(plants[i][1], plants[i][0])

plt.boxplot(values[1,:] - values[0,:], sym='')
plt.show()

with fiona.open(wr + "/20190717_count.shp") as src:

    with fiona.open(wr + "/20190717_count_diff2.shp", 'w', driver=src_driver, crs=src_crs, schema=src_schema) as dest:
        i = 0
        for d in src:
            d['properties']['diff'] = values[1, i] - values[0, i]
            i += 1
            dest.write(d)