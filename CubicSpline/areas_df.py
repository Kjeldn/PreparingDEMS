#import geopandas as gpd

from tkinter import filedialog
from tkinter import *
from pyqtree import Index
from shapely.geometry import mapping, MultiPoint, Point, Polygon
from shapely.strtree import STRtree
import re
import csv
from tqdm import tqdm
import scipy
import os
import geopandas as gpd

csv_dest = r"Z:\800 Operational\c07_hollandbean\Season evaluation\NoviFarm\areas.csv"

root = Tk()
voronoi_paths = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Season evaluation", title="Select voronoi polygons", parent=root, multiple=True)
plantcount_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Season evaluation", title="Select plant count", parent=root)


df = gpd.read_file(plantcount_path)
for i in df.index:
    if not df.loc[i,'geometry']:
        df = df.drop(i)
df = df.to_crs({'init': 'epsg:28992'})
bbox = MultiPoint([df.loc[i,'geometry'][0] if isinstance(df.loc[i,'geometry'], MultiPoint) else df.loc[i,'geometry'] for i in df.index]).bounds
keys = []
plants = df.loc[:, 'geometry']
polys_list = []
areas = []
pbar0 = tqdm(total=len(voronoi_paths), desc="reading voronoi polygons", position=0)
for p in voronoi_paths:
    df['area' + re.findall(r'\d+', p)[-1]] = None
    keys.append('area' + re.findall(r'\d+', p)[-1])
    f = gpd.read_file(p)
    f = f.to_crs({'init': 'epsg:28992'})
#    f = gpd.GeoDataFrame(f)
    f['isvalid'] = f.geometry.apply(lambda x: x.is_valid)
    f = f[(f['isvalid'] == True)]
    f = f.drop(columns = 'isvalid')
    polys_list.append(f.loc[:, 'geometry'])
    areas.append([0 for _ in range(len(plants))])
    pbar0.update(1)
pbar0.close()
    
pyq = Index(bbox=bbox)
for i, p  in enumerate(plants):
    pyq.insert(i, p.bounds)
    
convex_hull = Polygon([(p.x, p.y) for p in plants]).convex_hull

pbar = tqdm(total=len(voronoi_paths), position=0)
for j, f in enumerate(polys_list):
    pbar1 = tqdm(total=len(polys_list[j]), position=0)
    for p in f:
        if (p.is_valid and p.within(convex_hull)) or (not p.is_valid and p.buffer(0).within(convex_hull)):
            bbox = p.bounds
            intersected = pyq.intersect(bbox)
            if intersected:
                for k in intersected:
                    areas[j][k] = p.area / len(intersected)
        pbar1.update(1)
    pbar.update(1)
    pbar1.close()
pbar.close()

#with open(csv_dest, "w") as file:
#    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
#    writer.writerow(['x','y'] + keys)
#    for i in range(len(plants)):
#        p = plants[i] if isinstance(plants[i], Point) else plants[i][0]
#        writer.writerow([p.x, p.y] + [areas[j][i] for j in range(len(keys))])
        
#%%
import matplotlib.pyplot as plt
import datetime
import numpy as np
import divide_into_beds as dib
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union, linemerge
from matplotlib import cm
from matplotlib.colors import Normalize
import util_voronoi as util
import fiona
import itertools
import geopandas as gpd

dates_str = [re.findall(r'\d+', p)[-1] for p in voronoi_paths]
dates = [datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12])) for d in dates_str]

m = 20 #number of splitting lines parallel to the long side of the bed
n = 60 #number of splitting lines perpendicular to the long side of the bed
beds_points = dib.divide([(p.x, p.y) if isinstance(p, Point) else (p[0].x, p[0].y) for p in plants])
beds = [util.get_convex_hull(np.array(bed)) for bed in beds_points]
allpolys = []
f=1000
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
        
        if longest_line:
            longest_line = [(longest_line[0][0] + longest_length*np.cos(slope1), longest_line[0][1] + longest_length*np.sin(slope1)),
                            (longest_line[1][0] - longest_length*np.cos(slope1), longest_line[1][1] - longest_length*np.sin(slope1))]
            ret = []
            for i in range(1, 3*n + 1):
                point_to_add = [(i*longest_line[0][0] + (n + 1 -i) * longest_line[1][0])*(1/(n+1)), (i*longest_line[0][1] + (n + 1 -i) * longest_line[1][1])*(1/(n+1))]
                ret.append(point_to_add)
            
            ints = list(filter(lambda p : isinstance(p, MultiPoint), [LineString([(p[0]- f*np.cos(slope2),p[1] - f*np.sin(slope2)), (p[0] + f*np.cos(slope2),p[1] + f*np.sin(slope2))]).intersection(Polygon(points).exterior) for p in ret]))
    #        midpoints = [[[(i*ps[0].x + (m + 1 -i) * ps[1].x)*(1/(m+1)), (i*ps[0].y + (m + 1 -i) * ps[1].y)*(1/(m+1))] for ps in [ints[int(n/3)], ints[int(2*n/3)]]] for i in range(1, m + 1)]
            midpoints = [(0.5*ps[0].x + 0.5*ps[1].x, 0.5*ps[0].y + 0.5*ps[1].y) for ps in ints]
            polys = []
            lines1 = [LineString([(mp[0]- f*np.cos(slope1), mp[1] - f*np.sin(slope1)), (mp[0] + f*np.cos(slope1), mp[1] + f*np.sin(slope1))]) for mp in midpoints]
            lines2 = [LineString([(p[0]- f*np.cos(slope2),p[1] - f*np.sin(slope2)), (p[0] + f*np.cos(slope2),p[1] + f*np.sin(slope2))]) for p in ret]
            for p in polygonize(unary_union(linemerge(lines1 + lines2  + [LineString(bed.exterior.coords)]))):
                polys.append(p)
            allpolys += polys

poly_values = [[] for i in range(len(allpolys))]
for i, poly in enumerate(allpolys):
    for j in pyq.intersect(poly.bounds):
        try:
            if plants[j].within(poly) and areas[0][j] != 0:
                poly_values[i].append(areas[0][j])
        except:
            pass
     
m = np.mean(list(itertools.chain.from_iterable(poly_values)))
std2 = 2*np.std(list(itertools.chain.from_iterable(poly_values)))
for i in range(len(poly_values)):
    for j in range(len(poly_values[i]) -1, -1, -1):
        if abs(poly_values[i][j] - m) > std2:
            del poly_values[i][j]
            
min_mean = min([np.std(v) if v else 100 for v in poly_values])
max_mean = max([np.std(v) if v else -100 for v in poly_values])

for i, poly in enumerate(allpolys):
    if poly_values[i]:
        plt.fill(*poly.exterior.xy, c=cm.get_cmap('viridis')((np.std(poly_values[i]) - min_mean)/(max_mean-min_mean)))
        
sm = cm.ScalarMappable(norm=Normalize(min_mean, max_mean), cmap='viridis')
sm.set_array([])
plt.colorbar(sm)
plt.show()

rows = []

for i in range(len(poly_values)):
    if len(poly_values[i]) > 0:
        rows.append([np.std(poly_values[i]), allpolys[i]])
    
df = gpd.GeoDataFrame(rows,columns=['std','geometry'])
df.crs = {'init': 'epsg:28992'}

df.to_file(voronoi_paths[0].split(".")[0] + "_std.shp")

#%%
import copy
import numpy as np
means = []
for a in areas:
    ac = copy.deepcopy(a)
    m = np.mean(ac)
    std2 = 2*np.std(ac)
    for i in range(len(ac) -1, -1, -1):
        if abs(ac[i] - m) > std2:
            del ac[i]
    means.append(np.mean(ac))
    
plt.plot(means)
