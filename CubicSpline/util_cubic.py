import gdal
import numpy as np
from shapely.geometry import Polygon, LinearRing, Point
import numpy as np

class Plane:
    def __init__(self, array, gt):
        self.array = array
        self.gt = gt
        self.vcoord = np.vectorize(self.getCoordByIndices2)
        
    def getIndicesByCoord(self, cx, cy):
        return int(abs(np.floor((cx - self.gt[3])/self.gt[5]))), int(abs(np.floor((cy - self.gt[0])/self.gt[1])))
    
    def getCoordByIndices(self, x, y):
        return self.gt[3] + x * self.gt[5], self.gt[0] + y * self.gt[1] 
    
    def getCoordByIndices2(self, xy):
        return self.gt[3] + xy[0] * self.gt[5], self.gt[0] + xy[1] * self.gt[1] 
        
    def getMeanValueAt(self, x, y, k_size = 3):
        ptl = Point(int(abs(np.floor((x - self.gt[3])/self.gt[5]))), int(abs(np.floor((y - self.gt[0])/self.gt[1]))))
        sum_k = 0
        for i in range(ptl.x - int((k_size - 1)/2), ptl.x + int((k_size - 1)/2) + 1):
            for j in range(ptl.y - int((k_size - 1)/2), ptl.y + int((k_size - 1)/2) + 1):
                sum_k += self.array[i][j]
        return sum_k/(k_size**2)
    
    def getMaxValueAt(self, x, y, k_size = 7):
        xi, yi = self.getIndicesByCoord(x, y)
        return np.amax(self.array[xi - int((k_size - 1)/2):xi + int((k_size - 1)/2) + 1, yi - int((k_size - 1)/2):yi + int((k_size - 1)/2) + 1])
    
    def getMaxValueCoordAt(self, x, y, k_size = 7):
        xc, yc = self.getIndicesByCoord(x, y)
        maxValue, mx, my = 0
        for i in range(xc - int((k_size - 1)/2), xc + int((k_size - 1)/2) + 1):
            for j in range(yc - int((k_size - 1)/2), yc + int((k_size - 1)/2) + 1):
                if self.array[i][j] > maxValue:
                    maxValue = self.array[i][j]
                    mx = i
                    my = j
        return mx, my
    
    def maskValuesInNeighborhood(self, x, y, k_size = 7):
        xc, yc = self.getIndicesByCoord(x, y)
        self.array[xc - int((k_size - 1)/2):xc + int((k_size - 1)/2) + 1, yc - int((k_size - 1)/2):yc + int((k_size - 1)/2) + 1] = np.ones((k_size, k_size))
    
def create_tiff(array, gt, projection, dest: str, gdt_type = gdal.GDT_Float32):
    driver = gdal.GetDriverByName('GTiff')
    tiff = driver.Create(dest, array.shape[1], array.shape[0], 1, gdt_type, options=['COMPRESS=LZW'])
    tiff.SetGeoTransform(gt)
    tiff.SetProjection(projection)
    tiff.GetRasterBand(1).WriteArray(array)
    tiff.GetRasterBand(1).FlushCache()
    tiff = None
    
def get_convex_hull(plants):
    poly = Polygon(zip(plants[:,0], plants[:,1]))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    return polygon

def getMask(array, plants, gt, k_size = 15):
    if k_size % 2 == 0:
        raise ValueError("k_size should be odd")
    mask = np.zeros(array.shape)
    plane = Plane(mask, gt)
    
    for p in plants:
        if p:
            plane.maskValuesInNeighborhood(p[1], p[0], k_size)

    return plane.array.astype(np.uint8)