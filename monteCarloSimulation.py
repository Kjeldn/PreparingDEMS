# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:31:57 2019

@author: wytze
"""

import gdal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #shows as unused but is needed for surf plot
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.sparse import csr_matrix
from typing import List
from scipy import interpolate
import skgstat as skg

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def dist(self, p):
        return np.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)
        
class Polygon:
    EXTREME = 100000000
    def __init__(self, vs: List[Point]):
        self.vs = vs
    
    """Given three colinear points p, q, r, the function checks if q lies on line segment pr
    
    :param p: start of line segment pr
    :param q: the point to check
    :param r: end of line segment pr
    :returns: True if q on line segment pr
    """
    def on_segment(self, p, q, r):
        if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
            return True
        return False
    
    """Find orientation of ordered triplet (p, q, r)
    0 --> p, q, r are colinear
    1 --> Clockwise
    2 --> Counter clockwise
    
    :param p, q, r: points of which the orientation is checked
    :returns: orientation of the given points
    """
    def orientation(self, p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if (val == 0): 
            return 0
        return 1 if val > 0 else 2
    
    """Check if line segment p1q1 and p2q2 intersect
    
    :param p1, q1, p1, p2: the points of the line segments which are checked for intersection
    :returns: True if p1q1 and p2q2 intersect
    """
    def do_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
        
        if (o1 != o2 and o3 != o4):
            return True
        
        if (o1 == 0 and self.on_segment(p1, p2, q1)):
            return True
        
        if (o2 == 0 and self.on_segment(p1, q2, q1)):
            return True
        
        if (o3 == 0 and self.on_segment(p2, p1, q2)):
            return True
        
        if (o4 == 0 and self.on_segment(p2, q1, q2)):
            return True
        
        return False
    
    """Check if point p is inside the n-sized polygon given by points vs
    
    :param vs: list of points which make up the polygon
    :param n: number of vertices of the polygon
    :param p: the point to be checked
    :returns: True if p lies inside the polygon
    """
    def is_inside_polygon(self, p: Point):
        if len(self.vs) < 3:
            return False
        
        extreme = Point(self.EXTREME, p.y)
        
        count = 0
        i = 0
        while True:
            next_i = (i + 1) % len(self.vs)
            
            if self.do_intersect(self.vs[i], self.vs[next_i], p, extreme):
                if self.orientation(self.vs[i], p, self.vs[next_i]) == 0:
                    return self.on_segment(self.vs[i], p, self.vs[next_i])
                count += 1
                
            i = next_i
            if i == 0:
                break
        
        return count % 2 == 1
    
def create_tiff(array, gt, projection, dest: str):
    driver = gdal.GetDriverByName('GTiff')
    tiff = driver.Create(dest, array.shape[1], array.shape[0], 1, gdal.GDT_Int16)
    tiff.SetGeoTransform(gt)
    tiff.SetProjection(projection)
    tiff.GetRasterBand(1).WriteArray(array)
    tiff.GetRasterBand(1).FlushCache()
    tiff = None
    
#%% Monte Carlo simulation (tulips)
file = gdal.Open("C:/Users/wytze/OneDrive/Documents/vanBoven/Tulips/DEM/Achter_de_rolkas-20190420-DEM.tif")
ridges = gdal.Open("C:/Users/wytze/OneDrive/Documents/vanBoven/Tulips/DEM/crop_top.tif")

ridges_array = ridges.GetRasterBand(1).ReadAsArray()

band = file.GetRasterBand(1)
array = band.ReadAsArray()
projection = file.GetProjection()
gt = file.GetGeoTransform()
xsize = band.XSize
ysize = band.YSize

array[array == np.amin(array)] = 0

x1 = (52.27873864, 4.53404786) #top-left
x2 = (52.27856326, 4.53438642) #top-right
x3 = (52.27813579, 4.53389075) #bottom-right
x4 = (52.27825880, 4.53365449) #bottom-left

x1_i = Point(int(abs(np.floor((x1[0] - gt[3])/gt[5]))), int(abs(np.floor((x1[1] - gt[0])/gt[1]))))
x2_i = Point(int(abs(np.floor((x2[0] - gt[3])/gt[5]))), int(abs(np.floor((x2[1] - gt[0])/gt[1]))))
x3_i = Point(int(abs(np.floor((x3[0] - gt[3])/gt[5]))), int(abs(np.floor((x3[1] - gt[0])/gt[1]))))
x4_i = Point(int(abs(np.floor((x4[0] - gt[3])/gt[5]))), int(abs(np.floor((x4[1] - gt[0])/gt[1]))))

poly = Polygon([x1_i, x2_i, x3_i, x4_i])

data = []
for i in range(0, ysize, 50):
    for j in range(0, xsize, 50):
        if ridges_array[i][j] == 0 and poly.is_inside_polygon(Point(i,j)):
            data.append([i, j, array[i][j]])

data = np.array(data)
mean = np.mean(data[:, 2])
std = np.std(data[:, 2])

grid_x, grid_y = np.mgrid[0:ysize, 0:xsize]

N = 100
sims = np.zeros((ysize, xsize))
for i in range(N):
    data_s = np.zeros(data.shape)
    for j in range(len(data)):
        data_s[j, 0] = data[j, 0]
        data_s[j, 1] = data[j, 1]
        data_s[j, 2] = np.random.normal(data[j][2], std)
        
    sims += interpolate.griddata(data_s[:, 0:2], data_s[:, 2], (grid_x, grid_y), method="linear")
    
mean_sims = np.zeros((ysize, xsize))
for i in range(ysize):
    for j in range(xsize):
        if sims[i, j]:
            mean_sims[i][j] = sims[i,j]/N
    
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, mean_sims)
plt.show()

create_tiff(10 * (array - mean_sims), gt, projection, 'montecarlo_tulips.tif')

#%% Monte Carlo simulation (tulips)
file = gdal.Open("C:/Users/wytze/20190603_modified.tif")
ridges = gdal.Open("C:/Users/wytze/crop_top.tif")

ridges_array = ridges.GetRasterBand(1).ReadAsArray()

band = file.GetRasterBand(1)
array = band.ReadAsArray()
projection = file.GetProjection()
gt = file.GetGeoTransform()
xsize = band.XSize
ysize = band.YSize

array[array < 0] = 0

x1 = (51.73861232, 4.38036677) #top-left
x2 = (51.73829906, 4.38376774) #top-right
x3 = (51.73627495, 4.38329311) #bottom-right
x4 = (51.73661151, 4.37993097) #bottom-left

x1_i = Point(int(abs(np.floor((x1[0] - gt[3])/gt[5]))), int(abs(np.floor((x1[1] - gt[0])/gt[1]))))
x2_i = Point(int(abs(np.floor((x2[0] - gt[3])/gt[5]))), int(abs(np.floor((x2[1] - gt[0])/gt[1]))))
x3_i = Point(int(abs(np.floor((x3[0] - gt[3])/gt[5]))), int(abs(np.floor((x3[1] - gt[0])/gt[1]))))
x4_i = Point(int(abs(np.floor((x4[0] - gt[3])/gt[5]))), int(abs(np.floor((x4[1] - gt[0])/gt[1]))))

poly = Polygon([x1_i, x2_i, x3_i, x4_i])

data = []
for i in range(0, ysize, 50):
    for j in range(0, xsize, 50):
        if ridges_array[i][j] == 0 and poly.is_inside_polygon(Point(i,j)):
            data.append([i, j, array[i][j]])

data = np.array(data)
mean = np.mean(data[:, 2])
std = np.std(data[:, 2])

grid_x, grid_y = np.mgrid[0:ysize, 0:xsize]

N = 200
sims = np.zeros((ysize, xsize))
for i in range(N):
    data_s = np.zeros(data.shape)
    for j in range(len(data)):
        data_s[j, 0] = data[j, 0]
        data_s[j, 1] = data[j, 1]
        data_s[j, 2] = data[j, 2] + std * np.sqrt(-2 * np.log(1 - np.random.uniform(0.0, 0.99))) * np.cos(2 * np.pi * np.random.uniform(0.0, 0.99))
        
    sims += interpolate.griddata(data_s[:, 0:2], data_s[:, 2], (grid_x, grid_y), method="linear")
    print((i * 100)/N, '%')
    
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, sims/N)
plt.show()

create_tiff(10 * (array - sims/N), gt, projection, 'montecarlo_spruiten.tif')