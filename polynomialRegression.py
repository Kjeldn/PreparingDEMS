# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:12:31 2019

@author: wytze
"""

import gdal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #shows as unused but is needed for surf plot
import matplotlib.pyplot as plt
from scipy import stats


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

#%% Get the difference between a smoothed matrix and the real matrix
# =============================================================================
# smoothed_array = ndimage.gaussian_filter(array, sigma=20, order=0)
# 
# new_array = np.subtract(array, smoothed_array)
# 
# driver = gdal.GetDriverByName('GTiff')
# tiff = driver.Create('smoothed_array_diff.tif', xsize, ysize, 1, gdal.GDT_Int16)
# tiff.SetGeoTransform(gt)
# tiff.SetProjection(projection)
# #tiff.GetRasterBand(1).WriteArray(20 * new_array)
# tiff.GetRasterBand(1).WriteArray(smoothed_array)
# tiff.GetRasterBand(1).FlushCache()
# tiff = None
# =============================================================================

#%% Calculate the local minima in the oscillating lines in the matrix and assume the straight lines between minima are paralel to the ground
# =============================================================================
# for k in range(xsize):
#     first_zeros = []
#     last_zeros = []
#     b = 1.0 * ridges_array[:,k]
#     a = np.zeros(b.size)
#     last_zero = 0
#     i = last_zero
#     for i in range(b.size - 1):
#         if b[i] != b[i + 1] and b[i] == -1:
#             first_zeros.append(i)
#         elif b[i] != b[i + 1] and b[i] == 0:
#             last_zeros.append(i + 1)
#             
#     first_zeros = np.array(first_zeros)
#     last_zeros = np.array(last_zeros)
#     
#     for i in range(first_zeros.size - 1):
#         deltax = last_zeros[i] - first_zeros[i]
#         deltay = array[last_zeros[i], k] - array[first_zeros[i], k]
#         if deltax != 0:
#             for j in range(first_zeros[i], last_zeros[i + 1]):
#                 a[j] = array[j, k] - ((j - first_zeros[i]) * (deltay/deltax) + array[first_zeros[i], k])
#     array[:, k] = a.transpose()
#     
# array[np.abs(array) > 1] = 0
# driver = gdal.GetDriverByName('GTiff')
# tiff = driver.Create('ridges_diff.tif', xsize, ysize, 1, gdal.GDT_Int16)
# tiff.SetGeoTransform(gt)
# tiff.SetProjection(projection)
# tiff.GetRasterBand(1).WriteArray(20 * array)
# tiff.GetRasterBand(1).FlushCache()
# tiff = None
# =============================================================================

#%% Get edges of all crops
# =============================================================================
# def on_boundary(i, j, X):
#     for a in range(i - 1, i + 1):
#         for b in range(j - 1, j + 1):
#             if a >= 0 and b >= 0 and a < ysize and b < xsize and X[a, b] != X[i, j]:
#                 return True
#     
# X = np.zeros(array.shape)
# for i in range(ysize):
#     for j in range(xsize):
#         if on_boundary(i, j, ridges_array):
#             X[i, j] = 1
# 
# driver = gdal.GetDriverByName('GTiff')
# tiff = driver.Create('ridges_diff.tif', xsize, ysize, 1, gdal.GDT_Int16)
# tiff.SetGeoTransform(gt)
# tiff.SetProjection(projection)
# tiff.GetRasterBand(1).WriteArray(X)
# tiff.GetRasterBand(1).FlushCache()
# tiff = None
# =============================================================================

#%% Get polynomial fit of some points in the array and make a new tiff which is the difference between the original array and the polynomial fit
x1 = (51.73861232, 4.38036677) #top-left
x2 = (51.73829906, 4.38376774) #top-right
x3 = (51.73627495, 4.38329311) #bottom-right
x4 = (51.73661151, 4.37993097) #bottom-left

x1_i = (abs(np.floor((x1[0] - gt[3])/gt[1])), abs(np.floor((x1[1] - gt[0])/gt[5])))
x2_i = (abs(np.floor((x2[0] - gt[3])/gt[1])), abs(np.floor((x2[1] - gt[0])/gt[5])))
x3_i = (abs(np.floor((x3[0] - gt[3])/gt[1])), abs(np.floor((x3[1] - gt[0])/gt[5])))
x4_i = (abs(np.floor((x4[0] - gt[3])/gt[1])), abs(np.floor((x4[1] - gt[0])/gt[5])))

vs = [x1_i, x2_i, x3_i, x4_i]

xmin = 0
xmax = 7000
ymin = 0
ymax = 12000

xstep = 200
ystep = 200

x = []

"""Given three colinear points p, q, r, the function checks if q lies on line segment pr

:param p: start of line segment pr
:param q: the point to check
:param r: end of line segment pr
:returns: True if q on line segment pr
"""
def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

"""Find orientation of ordered triplet (p, q, r)
0 --> p, q, r are colinear
1 --> Clockwise
2 --> Counter clockwise

:param p, q, r: points of which the orientation is checked
:returns: orientation of the given points
"""
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if (val == 0): 
        return 0
    return 1 if val > 0 else 2

"""Check if line segment p1q1 and p2q2 intersect

:param p1, q1, p1, p2: the points of the line segments which are checked for intersection
:returns: True if p1q1 and p2q2 intersect
"""
def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if (o1 != o2 and o3 != o4):
        return True
    
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True
    
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True
    
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True
    
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False

"""Check if point p is inside the n-sized polygon given by points vs

:param vs: list of points which make up the polygon
:param n: number of vertices of the polygon
:param p: the point to be checked
:returns: True if p lies inside the polygon
"""
def is_inside_polygon(vs, n, p):
    if n < 3:
        return False
    
    extreme = (10000000, p[1])
    
    count = 0
    i = 0
    while True:
        next_i = (i + 1) % n
        
        if do_intersect(vs[i], vs[next_i], p, extreme):
            if orientation(vs[i], p, vs[next_i]) == 0:
                return on_segment(vs[i], p, vs[next_i])
            count += 1
            
        i = next_i
        if i == 0:
            break
    
    return count % 2 == 1    
    
# create list of points inside the field to get the fit over
for i in range(int((xmax - xmin)/xstep)):
    for j in range(int((ymax - ymin)/ystep)):
        if ridges_array[i, j] == 0 and is_inside_polygon(vs, 4, (xmin + xstep * i, ymin + ystep * j)):
            x.append((xmin + xstep * i, ymin + ystep * j))
            
order = 3
n_terms = int(((order + 1) * (order + 2)) / 2)

# Create matrix which describes (1, x, x^2, x*y, y, y^2) and vector for z-values in all points of x
z = np.zeros(len(x))
X = np.zeros((len(x), n_terms))
# =============================================================================
# for i in range(len(x)):
#     z[i] = array[x[i][0], x[i][1]]
#     
#     X[i, 0] = 1
#     X[i, 1] = x[i][0]
#     X[i, 2] = x[i][0]**2
#     X[i, 3] = x[i][0] * x[i][1]
#     X[i, 4] = x[i][1]
#     X[i, 5] = x[i][1]**2
#     if order == 3:
#         X[i, 6] = x[i][0]**3
#         X[i, 7] = x[i][0]**2*x[i][1]
#         X[i, 8] = x[i][0]*x[i][1]**2
#         X[i, 9] = x[i][1]**3
# =============================================================================
    
for i in range(len(x)):
    z[i] = array[x[i][0], x[i][1]]
    
    X[i, 0] = 1
    a = 1
    for k in range(1, order + 1):
        for l in range(k + 1):
            X[i, a] = x[i][0]**l * x[i][1]**(k - l)
            a += 1

# Create function z = b[0] + b[1]*x + b[2]*x^2 + b[3]*x*y + b[4]*y + b[5]*y^2 + b[6]*x^3 + b[7]*x^2*y + b[8]*x*y^2 + b[9]*y^3
# b = (X^T*X)^-1 X^T * z
XT = X.transpose()
b = np.linalg.inv(np.matmul(XT, X)) @ (XT @ z)

Z = np.zeros(array.shape)
for i in range(ysize):
    for j in range(xsize):
        a = 0
        for k in range(order + 1):
            for l in range(k + 1):
                Z[i, j] += b[a] * i**l * j**(k - l)
                a += 1
# =============================================================================
# for i in range(ysize):
#     for j in range(xsize):
#         Z[i][j] = b[0] + i * b[1] + i **2 * b[2] + i * j * b[3] + j * b[4] + j * j * b[5]
#         if order == 3:
#             Z[i][j] += i**3 * b[6] + i**2 * j * b[7] + i * j**2 * b[8] + j**3 * b[9]
# =============================================================================

# graphical representation of the polynome        
xx = np.arange(0, xsize, 1)
yy = np.arange(0, ysize, 1)

xx, yy = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, Z)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, Z - array)
plt.show()

# create tiff file with difference between polynome and original matrix
driver = gdal.GetDriverByName('GTiff')
tiff = driver.Create('poly_regres.tif', xsize, ysize, 1, gdal.GDT_Int16)
tiff.SetGeoTransform(gt)
tiff.SetProjection(projection)
tiff.GetRasterBand(1).WriteArray(20 * (array - Z))
tiff.GetRasterBand(1).FlushCache()
tiff = None

#%% Goodness of fit
chi_squared = 0
e = np.zeros(len(x))
for i in range(len(x)):
    e[i] = Z[x[i][0], x[i][1]]

df = len(x) - 1

print(stats.chisquare(z, e, 5))

