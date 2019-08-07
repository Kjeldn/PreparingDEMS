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
from scipy.sparse import csr_matrix
from typing import List
from scipy import interpolate
import skgstat as skg
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

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

xstep = 2
xmax = 3000
x = range(xstep, xmax, xstep)
gamma = np.zeros(len(x))

for a in x:
    sum0 = 0
    V = 0
    for i in range(0, ysize, a):
        for j in range(0, xsize, a):
            if i + a < ysize and array[i][j] != 0 and array[i][j] and array[i + a][j] and ridges_array[i][j] == 0 and ridges_array[i + a][j] == 0:
                sum0 += (array[i][j] - array[i + a][j])**2
                V += 1
    gamma[x.index(a)] = sum0 /(2 * V) if V != 0 else 0

deltah = 500
h_b = np.arange(0, xmax, deltah)
gamma_b = np.zeros(len(h_b))
for i in range(len(gamma_b)):
    sum1 = 0
    n1 = 0
    for j in range(len(gamma)):
        if x[j] > h_b[i] - deltah/2 and x[j] < h_b[i] + deltah/2:
            n1 += 1
            sum1 += gamma[i]
    gamma_b[i] = sum1 / n1

plt.plot(h_b, gamma_b)
coeff = np.polyfit(h_b, gamma_b, 2)
gamma_fit = coeff[2] + coeff[1] * x + coeff[0] * np.square(x)

n = coeff[2]
s = np.amax(gamma_fit) - n
r = float(np.argmax(gamma_fit))

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

def v_func(l, h):
    ret = np.zeros(h.shape)
    if len(h.shape) == 2:
        for i in range(h.shape[0]):    
            for j in range(h.shape[1]):      
                if h[i][j] < r:
                    ret[i][j] = n + s*(1.5 * (h[i][j] / r) - 0.5 * (h[i][j]/r)**3)
                else:
                    ret[i][j] = n + s
    else:
        for i in range(h.shape[0]):         
            if h[i] < r:
                ret[i] = n + s*(1.5 * (h[i] / r) - 0.5 * (h[i]/r)**3)
            else:
                ret[i] = n + s
    return ret

#%% kriging framework (tulips)
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

point_cloud = []

for i in range(0, ysize, 50):
    for j in range(0, xsize, 50):
        if array[i][j] != 0:
            point_cloud.append([float(i), float(j), array[i][j]])
            
point_cloud = np.array(point_cloud)

X = np.zeros((len(point_cloud), 6))
z = np.zeros((len(point_cloud), 1))
for i in range(len(point_cloud)):
    X[i, 0] = 1
    X[i, 1] = point_cloud[i][0]
    X[i, 2] = point_cloud[i][1]
    X[i, 3] = point_cloud[i][0]**2
    X[i, 4] = point_cloud[i][0] * point_cloud[i][1]
    X[i, 5] = point_cloud[i][1]**2
    z[i] = point_cloud[1][2]
    
b = np.linalg.inv(X.T @ X) @ X.T @ z

array_wt = np.zeros(array.shape)

for i in range(0, ysize):
    for j in range(0, xsize):
        if array[i][j] != 0:
            array_wt[i][j] =  array[i][j] - (b[0] + i * b[1] + j * b[2] + i**2*b[3] + i * j * b[4] + j**2 * b[5])
            
data = []

for i in range(0, ysize, 50):
    for j in range(0, xsize, 50):
        if array_wt[i][j] != 0 and ridges_array[i][j] == 0 and poly.is_inside_polygon(Point(i, j)):
            data.append([float(i), float(j), array_wt[i][j]])
            
data = np.array(data)

delta = 10
OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='spherical', verbose=True, enable_plotting=True)
z, ss = OK.execute('grid', np.arange(0, float(xsize), delta), np.arange(0, float(ysize), delta))

grid_x_sparse, grid_y_sparse = np.mgrid[0:float(ysize):delta, 0:float(xsize):delta]

points = []
values = []
for i in range(0, ysize, delta):
    for j in range(0, xsize, delta):
        points.append([i, j])
        values.append(z[int(i/delta), int(j/delta)])
        
grid_x, grid_y = np.mgrid[0:ysize, 0:xsize]

e = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, e)
plt.show()

create_tiff(10 * (array - e), gt, projection, 'kriging.tif')

#%% kriging framework (spruiten)
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

# =============================================================================
# point_cloud = []
# 
# for i in range(0, ysize, 50):
#     for j in range(0, xsize, 50):
#         if array[i][j] != 0 and poly.is_inside_polygon(Point(i, j)):
#             point_cloud.append([float(i), float(j), array[i][j]])
#             
# point_cloud = np.array(point_cloud)
# 
# X = np.zeros((len(point_cloud), 6))
# z = np.zeros((len(point_cloud), 1))
# for i in range(len(point_cloud)):
#     X[i, 0] = 1
#     X[i, 1] = point_cloud[i][0]
#     X[i, 2] = point_cloud[i][1]
#     X[i, 3] = point_cloud[i][0]**2
#     X[i, 4] = point_cloud[i][0] * point_cloud[i][1]
#     X[i, 5] = point_cloud[i][1]**2
#     z[i] = point_cloud[1][2]
#     
# b = np.linalg.inv(X.T @ X) @ X.T @ z
# 
# array_wt = np.zeros(array.shape)
# =============================================================================
array_wt = array

# =============================================================================
# for i in range(0, ysize):
#     for j in range(0, xsize):
#         if array[i][j] != 0:
#             array_wt[i][j] =  array[i][j] - (b[0] + i * b[1] + j * b[2] + i**2*b[3] + i * j * b[4] + j**2 * b[5])
# =============================================================================
            
data = []

for i in range(0, ysize, 25):
    for j in range(0, xsize, 25):
        if array_wt[i][j] != 0 and ridges_array[i][j] == 0 and poly.is_inside_polygon(Point(i, j)):
            data.append([float(i), float(j), array_wt[i][j]])
            
data = np.array(data)

delta = 20
UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='spherical', verbose=True, enable_plotting=True)
z, ss = UK.execute('grid', np.arange(0, float(xsize), delta), np.arange(0, float(ysize), delta))

grid_x_sparse, grid_y_sparse = np.mgrid[0:float(ysize):delta, 0:float(xsize):delta]

points = []
values = []
for i in range(0, ysize, delta):
    for j in range(0, xsize, delta):
        points.append([i, j])
        values.append(z[int(i/delta), int(j/delta)])
        
grid_x, grid_y = np.mgrid[0:ysize, 0:xsize]

e = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, e)
plt.show()

create_tiff(10 * (array - e), gt, projection, 'kriging.tif')

#%% kriging algorithm
in_field = np.zeros(array.shape)
for i in range(0, ysize):
    for j in range(0, xsize):
        if poly.is_inside_polygon(Point(i,j)):
            in_field[i][j] = 1

C = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        C[i][j] = corr(Point(data[i][0], data[i][1]).dist(Point(data[j][0], data[j][1])))

C = np.matrix(C).T
Cinv = np.linalg.inv(C)
Cinv_sum = float(np.ones((1, len(data))) @ Cinv @ np.ones((len(data), 1)))

delta = 5
e = np.zeros(array.shape)
for i in range(len(data)):
    e[int(data[i][0])][int(data[i][1])] = - data[i][2]
    
percent = 0

e_sparse = np.zeros((int(array.shape[0] / delta), int(array.shape[1] / delta)))

for i in range(e_sparse.shape[0]):
    for j in range(e_sparse.shape[1]):
        if in_field[int(i * delta)][int(j * delta)] == 1:
            D = np.zeros((len(data), 1))
            for k in range(len(data)):
                D[k] = corr(Point(i * delta, j * delta).dist(Point(data[k][0], data[k][1])))
            l = float(D.T @ Cinv @ np.ones((len(data), 1)) - 1) / Cinv_sum
            w = Cinv @ (D - l * np.ones((len(data), 1)))
            sum0 = 0
            for k in range(len(data)):
                sum0 += w[k] * data[k][2]
            e[i][j] = sum0
    if np.floor((i/e_sparse.shape[0]) * 100) > percent:
        percent += 1
        print(percent,"%")
        
points = []
values = []
for i in range(0, ysize, delta):
    for j in range(0, xsize, delta):
        points.append([i, j])
        values.append(e[int(i/delta), int(j/delta)])
        
grid_x, grid_y = np.mgrid[0:ysize, 0:xsize]

e = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, e)
plt.show()

create_tiff(array - e, gt, projection, 'kriging.tif')

