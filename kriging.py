# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:50:56 2019

@author: wytze
"""

import gdal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #shows as unused but is needed for surf plot
import matplotlib.pyplot as plt
from scipy import stats, signal
from typing import List
from scipy import interpolate
import skgstat as skg

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

#%% Polygon and Point class
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
    
#%% semivariogram
x1 = (52.27874252, 4.53404572) #top-left
x2 = (52.27856259, 4.53438068) #top-right
x3 = (52.27813766, 4.53389149) #bottom-right
x4 = (52.27826056, 4.53364649) #bottom-left

x1_i = Point(int(abs(np.floor((x1[0] - gt[3])/gt[1]))), int(abs(np.floor((x1[1] - gt[0])/gt[5]))))
x2_i = Point(int(abs(np.floor((x2[0] - gt[3])/gt[1]))), int(abs(np.floor((x2[1] - gt[0])/gt[5]))))
x3_i = Point(int(abs(np.floor((x3[0] - gt[3])/gt[1]))), int(abs(np.floor((x3[1] - gt[0])/gt[5]))))
x4_i = Point(int(abs(np.floor((x4[0] - gt[3])/gt[1]))), int(abs(np.floor((x4[1] - gt[0])/gt[5]))))

poly = Polygon([x1_i, x2_i, x3_i, x4_i])

xstep = 2
xmax = 400
x = range(xstep, xmax, xstep)
gamma = np.zeros(len(x))

for a in x:
    sum0 = 0
    V = 0
    for i in range(0, ysize, a):
        for j in range(0, xsize, a):
            if i + a < ysize and array[i][j] != 0 and array[i][j] and array[i + a][j] and ridges_array[i][j] == 0:
                sum0 += (array[i][j] - array[i + a][j])**2
                V += 1
    gamma[x.index(a)] = sum0 /(2 * V) if V != 0 else 0

for i in range(len(gamma) - 1, -1, -1):
    if gamma[i] > 0.15:
        gamma = np.delete(gamma, i)
        x = np.delete(x, i)
        
plt.plot(x, gamma)
coeff = np.polyfit(x, gamma, 2)
gamma_fit = coeff[2] + coeff[1] *x + coeff[0] * np.square(x)

s = np.amax(gamma_fit)
r = float(np.argmax(gamma_fit))
n = coeff[2]

vario = np.ones(len(x)) * (s + n)

for i in range(0, len(x)):
    if x[i] < r:
        vario[i] = n + s*(1.5 * (x[i] / r) - 0.5 * (x[i]/r)**3)
        
def corr(h):
    if h < r:
        return n + s*(1 - n + s*(1.5 * (h/r) - 0.5 * (h/r)**3))
    else:
        return 0
    
plt.plot(x, vario)
plt.show()

#%%
x1 = (52.27874252, 4.53404572) #top-left
x2 = (52.27856259, 4.53438068) #top-right
x3 = (52.27813766, 4.53389149) #bottom-right
x4 = (52.27826056, 4.53364649) #bottom-left

x1_i = Point(int(abs(np.floor((x1[0] - gt[3])/gt[1]))), int(abs(np.floor((x1[1] - gt[0])/gt[5]))))
x2_i = Point(int(abs(np.floor((x2[0] - gt[3])/gt[1]))), int(abs(np.floor((x2[1] - gt[0])/gt[5]))))
x3_i = Point(int(abs(np.floor((x3[0] - gt[3])/gt[1]))), int(abs(np.floor((x3[1] - gt[0])/gt[5]))))
x4_i = Point(int(abs(np.floor((x4[0] - gt[3])/gt[1]))), int(abs(np.floor((x4[1] - gt[0])/gt[5]))))

poly = Polygon([x1_i, x2_i, x3_i, x4_i])

vs = []
values = []

for i in range(0, ysize, 50):
    for j in range(0, xsize, 50):
        if poly.is_inside_polygon(Point(i,j)) and ridges_array[i,j] == 0:
            vs.append((i, j))
            values.append(array[i][j])
            
C = np.zeros((len(vs), len(vs)))
for i in range(len(vs)):
    for j in range(i, len(vs)):
        C[i][j] = corr(Point(vs[i][0], vs[i][1]).dist(Point(vs[j][0], vs[j][1])))

C = np.matrix(C).T

e = np.zeros(array.shape)
for i in range(len(vs)):
    e[vs[i][0]][vs[i][1]] = - array[vs[i][0]][vs[i][1]]

for i in range(ysize):
    for j in range(xsize):
        if poly.is_inside_polygon(Point(i, j)) and (i, j) not in vs:
            D = np.zeros((len(vs), 1))
            for k in range(len(vs)):
                D[k] = corr(Point(i, j).dist(Point(vs[k][0], vs[k][1])))
            l = (D.T @ np.linalg.inv(C) @ np.ones((len(vs), 1)) - 1) / (np.ones((1, len(vs))) @ np.linalg.inv(C) @ np.ones((len(vs), 1)))
            w = np.linalg.inv(C) @ (D - float(l) * np.ones((len(vs), 1)))
            sum0 = 0
            for k in range(len(vs)):
                sum0 += w[k] * array[vs[k][0]][vs[k][1]]
            e[i][j] = sum0
    print(100*(i/ysize))            

