import gdal
import util
import numpy as np
import fiona
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from tqdm import trange
from pyqtree import Index

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

# =============================================================================
# z = heights[:,1] - heights[:,0]
# plt.scatter(np.array(plants)[:,0], np.array(plants)[:,1], c=z, cmap="Reds")
# plt.show()
# =============================================================================

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

spindex = Index(bbox=(np.amin(np.array(plants)[:,0]), np.amin(np.array(plants)[:,1]), np.amax(np.array(plants)[:,0]), np.amax(np.array(plants)[:,1])))
for i,plant in enumerate(plants):
    spindex.insert({'obj': plant, 'index': i}, bbox=(plant[0], plant[1], plant[0], plant[1]))
    
fig = plt.figure()
for i, bed in enumerate(beds):
    plt.fill(*bed.exterior.xy, c=(np.mean(height_beds[i])/max_mean,0,0))
fig.show()

for k in trange(len(beds), desc="driehoeken", position=0):
    bed = beds[k]
    if bed.exterior:
        x,y = bed.exterior.xy   
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )
        alpha = np.linspace(0, 1, 50)
        x_regular, y_regular = fx(alpha), fy(alpha)
        tri = Delaunay(np.array([x_regular, y_regular]).T)
        
        tri_values = [[] for i in range(len(tri.simplices))]
        for j, s in enumerate(tri.simplices):
            for plant in spindex.intersect(Polygon(tri.points[s]).bounds):
                if Point(plant['obj']).within(Polygon(tri.points[s])):
                    tri_values[j].append(heights[plant['index']][1] - heights[plant['index']][0])
        
        min_mean = min([np.mean(v) if v else 100 for v in tri_values])
        max_mean = max([np.mean(v) if v else -100 for v in tri_values])
        for i,s in enumerate(tri.simplices):
            if tri_values[i]:
                plt.fill(tri.points[s][:,0], tri.points[s][:,1], c=((np.mean(tri_values[i]) - min_mean)/(max_mean-min_mean), 0, 0))
                
#%%
from shapely.geometry import LineString, LinearRing
allpolys = []
for bed in beds:
    if bed.exterior:
        points = list(bed.exterior.coords)
        
        for i in range(len(points) -2, 0, -1):
            if abs(np.arctan((points[i][1] - points[i - 1][1])/(points[i][0] - points[i - 1][0])) - np.arctan((points[i + 1][1] - points[i][1])/(points[i + 1][0] - points[i][0]))) < np.pi/8:
                del points[i]
                
        longest_line = None
        longest_length = 0
        for i in range(len(points)):
            if Point(points[i]).distance(Point(points[(i + 1) % len(points)])) > longest_length:
                longest_line = [points[i], points[(i + 1) % len(points)]]
                longest_length = Point(points[i]).distance(Point(points[(i + 1) % len(points)]))
                
        ret = []
        n = 20
        for i in range(1, n + 1):
            point_to_add = [(i*longest_line[0][0] + (n + 1 -i) * longest_line[1][0])*(1/(n+1)), (i*longest_line[0][1] + (n + 1 -i) * longest_line[1][1])*(1/(n+1))]
            ret.append(point_to_add)
            
        slope = np.arctan((longest_line[1][1] - longest_line[0][1])/(longest_line[1][0] - longest_line[0][0])) + np.pi/2
        ints = [LineString([(p[0]- 0.01*np.cos(slope),p[1] - 0.01*np.sin(slope)), (p[0] + 0.01*np.cos(slope),p[1] + 0.01*np.sin(slope))]).intersection(Polygon(points).exterior)[1] for p in ret]
        
        side1 = []
        side2 = []
        for p in points:
            if LinearRing([(p[0], p[1]), ints[0].coords[0], ints[0].coords[1]]).is_ccw:
                side1.append(p)
            else:
                side2.append(p)
                
        polys = []
        for i in range(len(ints)-1):
            polys.append(Polygon([ints[i].coords[0], ints[i].coords[1], ints[i+1].coords[1], ints[i+1].coords[0]]))
        
        polyunion = polys[0]
        for i in range(1, len(polys)):
            polyunion = polyunion.union(polys[i])
            
        diff_polys = sorted(Polygon(points).difference(polyunion), key=lambda p : p.area, reverse=True)
        polys.append(diff_polys[0])
        polys.append(diff_polys[1])
        allpolys += polys

poly_values = [[] for i in range(len(allpolys))]
for i, poly in enumerate(allpolys):
    for plant in spindex.intersect(poly.bounds):
        if Point(plant['obj']).within(poly):
            poly_values[i].append(heights[plant['index']][1] - heights[plant['index']][0])
            
min_mean = min([np.mean(v) if v else 100 for v in poly_values])
max_mean = max([np.mean(v) if v else -100 for v in poly_values])

for i, poly in enumerate(allpolys):
    if poly.values[i]:
        plt.fill(*poly.exterior.xy, c=((np.mean(poly_values[i]) - min_mean)/(max_mean-min_mean), 0, 0))