import gdal
import util_cubic as util
import numpy as np
import fiona
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, LinearRing, mapping
from shapely.ops import linemerge, unary_union, polygonize
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from pyqtree import Index
from matplotlib import cm
from matplotlib.colors import Normalize
import divide_into_beds as dib
import datetime
from tkinter import filedialog
from tkinter import *

root = Tk()
paths = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select dems", parent=root, multiple=True)
plants_count_path = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare", title="Select plant count", parent=root)
root.destroy()
planes = []
colormap = "viridis"

for path in paths:
    file = gdal.Open(path)
    gt = file.GetGeoTransform()
    projection = file.GetProjection()
    band = file.GetRasterBand(1)
    array = band.ReadAsArray()
    planes.append(util.Plane(array, gt))
    file = None

plants = []
with fiona.open(plants_count_path) as src:
    print(src.crs)
    for s in src:
        if s['geometry']:
            if s['geometry']['type'] == 'Point':
                plants.append(s['geometry']['coordinates'])
            elif s['geometry']['type'] == 'MultiPoint':
                plants.append(s['geometry']['coordinates'][0])

heights = np.zeros((len(plants), len(paths)))
for i in range(len(plants)):
    heights[i,:] = np.array([plane.getMaxValueAt(plants[i][1], plants[i][0], k_size=15) for plane in planes])
    
beds_points = dib.divide(plants)
beds = [util.get_convex_hull(np.array(bed)) for bed in beds_points]

height_beds = [[] for i in range(len(beds))]
for i, plant in enumerate(plants):
    for j, bed in enumerate(beds):
        if Point(plant).within(bed):
            height_beds[j].append(heights[i,1] - heights[i,0])
            
min_mean = min([np.mean(v) if v else 100 for v in height_beds])
max_mean = max([np.mean(v) if v else -100 for v in height_beds])

spindex = Index(bbox=(np.amin(np.array(plants)[:,0]), np.amin(np.array(plants)[:,1]), np.amax(np.array(plants)[:,0]), np.amax(np.array(plants)[:,1])))
for i,plant in enumerate(plants):
    spindex.insert({'obj': plant, 'index': i}, bbox=(plant[0], plant[1], plant[0], plant[1]))
#%%
dates_str = [p.split("_DEM")[0].split("-")[-1] for p in paths]
dates = [datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12])) for d in dates_str]
time_diffs = [(dates[i] - dates[0]).total_seconds() for i in range(len(dates))]
plt.plot(time_diffs, [np.median(heights[:,i]) for i in range(len(planes))])
plt.fill_between(time_diffs, [np.percentile(heights[:,i], 25) for i in range(len(planes))], [np.percentile(heights[:,i], 75) for i in range(len(planes))], color="cyan")
plt.show()
#%%
for a in range(len(paths) - 1):
    
    diff = (a, a+1) #height difference between path[diff[1]] and path[diff[0]]
    m = 20 #number of splitting lines parallel to the long side of the bed
    n = 60 #number of splitting lines perpendicular to the long side of the bed
    allpolys = []
    for bed in beds:
        if bed.exterior:
            points = list(bed.exterior.coords)
            
            for i in range(len(points) -2, 0, -1):
                if abs(np.arctan((points[i][1] - points[i - 1][1])/(points[i][0] - points[i - 1][0])) - np.arctan((points[i + 1][1] - points[i][1])/(points[i + 1][0] - points[i][0]))) < np.pi/8:
                    del points[i]
                    
            sortedLines = sorted([sorted([Point(points[i]), Point(points[i+1])], key=lambda a: a.x) for i in range(len(points) -1)], key=lambda a: a[0].distance(a[1]), reverse=True)
            slopes = []
            for l in sortedLines:
                try:
                    slopes.append(np.arctan((l[1].y - l[0].y)/(l[1].x - l[0].x)))
                except:
                    slopes.append(0)
            slope1 = 0
            slope2 = 0
            for s in slopes:
                if slope1 == 0:
                    slope1 = s
                elif slope2 == 0 and abs(s - slope1) > np.pi/8:
                    slope2 = s
                    
            longest_line = None
            longest_length = 0
            for i in range(len(points)):
                if Point(points[i]).distance(Point(points[(i + 1) % len(points)])) > longest_length:
                    longest_line = [points[i], points[(i + 1) % len(points)]]
                    longest_length = Point(points[i]).distance(Point(points[(i + 1) % len(points)]))
                    
            longest_line = [(longest_line[0][0] + longest_length*np.cos(slope1), longest_line[0][1] + longest_length*np.sin(slope1)),
                            (longest_line[1][0] - longest_length*np.cos(slope1), longest_line[1][1] - longest_length*np.sin(slope1))]
            ret = []
            for i in range(1, 3*n + 1):
                point_to_add = [(i*longest_line[0][0] + (n + 1 -i) * longest_line[1][0])*(1/(n+1)), (i*longest_line[0][1] + (n + 1 -i) * longest_line[1][1])*(1/(n+1))]
                ret.append(point_to_add)
            
            ints = [LineString([(p[0]- 0.01*np.cos(slope2),p[1] - 0.01*np.sin(slope2)), (p[0] + 0.01*np.cos(slope2),p[1] + 0.01*np.sin(slope2))]).intersection(Polygon(points).exterior) for p in ret]
            midpoints = [[[(i*ps[0].x + (m + 1 -i) * ps[1].x)*(1/(m+1)), (i*ps[0].y + (m + 1 -i) * ps[1].y)*(1/(m+1))] for ps in [ints[int(n/3)], ints[int(2*n/3)]]] for i in range(1, m + 1)]
            polys = []
            lines1 = [LineString([(mp[0][0]- 0.01*np.cos(slope1), mp[0][1] - 0.01*np.sin(slope1)), (mp[0][0] + 0.01*np.cos(slope1), mp[0][1] + 0.01*np.sin(slope1))]) for mp in midpoints]
            lines2 = [LineString([(p[0]- 0.01*np.cos(slope2),p[1] - 0.01*np.sin(slope2)), (p[0] + 0.01*np.cos(slope2),p[1] + 0.01*np.sin(slope2))]) for p in ret]
            for p in polygonize(unary_union(linemerge(lines1 + lines2  + [LineString(bed.exterior.coords)]))):    
                polys.append(p)
            allpolys += polys
    
    poly_values = [[] for i in range(len(allpolys))]
    for i, poly in enumerate(allpolys):
        for plant in spindex.intersect(poly.bounds):
            if Point(plant['obj']).within(poly):
                poly_values[i].append(heights[plant['index']][diff[1]] - heights[plant['index']][diff[0]])
                
    min_mean = min([np.mean(v) if v else 100 for v in poly_values])
    max_mean = max([np.mean(v) if v else -100 for v in poly_values])
    
# =============================================================================
#     for i, poly in enumerate(allpolys):
#         if poly_values[i]:
#             plt.fill(*poly.exterior.xy, c=cm.get_cmap(colormap)((np.mean(poly_values[i]) - min_mean)/(max_mean-min_mean)))
#             
#     sm = cm.ScalarMappable(norm=Normalize(min_mean, max_mean), cmap=colormap)
#     sm.set_array([])
#     plt.colorbar(sm)
#     plt.show()
# =============================================================================

    schema = {
        'geometry': 'Polygon',
        'properties': {'mean_height_diff': 'float'},
    }
    wd = "/".join(paths[0].split("/")[:-1])
    date1 = paths[a].split("_DEM")[0].split("-")[-1][4:8]
    date2 = paths[a+1].split("_DEM")[0].split("-")[-1][4:8]
    with fiona.open(wd + '/height_diff_polygons_' + date1 + "-" + date2 + ".gpkg", 'w', crs={'init': 'epsg:4326'}, driver='GPKG', schema=schema) as c:
        for i, p in enumerate(allpolys):
            if poly_values[i]:
                c.write({
                    'geometry': mapping(p),
                    'properties': {'mean_height_diff': np.mean(poly_values[i])},
                })