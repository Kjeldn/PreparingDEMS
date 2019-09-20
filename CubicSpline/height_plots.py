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

wd = r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants compare2"
paths = ["c01_verdonk-Wever west-201907170749_DEM-GR_cubic", 
         "c01_verdonk-Wever west-201907240724_DEM-GR_cubic",
         "c01_verdonk-Wever west-201908041528_DEM-GR_cubic"]
plants_count_path = "20190717_count"
planes = []
colormap = "viridis"

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
    
plt.plot(np.arange(len(planes)), [np.median(heights[:,i]) for i in range(len(planes))])
plt.fill_between(np.arange(len(planes)), [np.percentile(heights[:,i], 25) for i in range(len(planes))], [np.percentile(heights[:,i], 75) for i in range(len(planes))], color="cyan")
plt.show()
#%%
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
#%%
# =============================================================================
# z = heights[:,1] - heights[:,0]
# plt.scatter(np.array(plants)[:,0], np.array(plants)[:,1], c=z, cmap="Reds")
# plt.show()
# =============================================================================
#%%
plt.hist(heights[:,0],bins=1000,color=(1, 0, 0, 0.5))
plt.hist(heights[:,1],bins=1000,color=(0, 0, 1, 0.5))
plt.hist(heights[:,2],bins=1000,color=(1, 1, 0, 0.5))
plt.hist(heights[:,3],bins=1000,color=(0, 1, 0, 0.5))
plt.show()
#%% 
fig = plt.figure()
for i, bed in enumerate(beds):
    plt.fill(*bed.exterior.xy, c=cm.get_cmap(colormap)((np.mean(height_beds[i]) - min_mean)/(max_mean-min_mean)))
fig.show()
sm = cm.ScalarMappable(norm=Normalize(min_mean, max_mean), cmap=colormap)
sm.set_array([])
plt.colorbar(sm)
plt.show()
#%%
allsimplices = []
for k in range(len(beds)):
    bed = beds[k]
    if bed.exterior:
        x,y = bed.exterior.xy   
        distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
        distance = distance/distance[-1]
        fx, fy = interp1d(distance, x), interp1d(distance, y)
        alpha = np.linspace(0, 1, 50)
        x_regular, y_regular = fx(alpha), fy(alpha)
        tri = Delaunay(np.array([x_regular, y_regular]).T)
        allsimplices += [Polygon(tri.points[s]) for s in tri.simplices]
        
tri_values = [[] for i in range(len(allsimplices))]
for j, s in enumerate(allsimplices):
    for plant in spindex.intersect(s.bounds):
        if Point(plant['obj']).within(s):
            tri_values[j].append(heights[plant['index']][1] - heights[plant['index']][0])

min_mean = min([np.mean(v) if v else 100 for v in tri_values])
max_mean = max([np.mean(v) if v else -100 for v in tri_values])
for i, s in enumerate(allsimplices):
    if tri_values[i]:
        plt.fill(*s.exterior.xy, c=cm.get_cmap(colormap)((np.mean(tri_values[i]) - min_mean)/(max_mean-min_mean)))
        
sm = cm.ScalarMappable(norm=Normalize(min_mean, max_mean), cmap=colormap)
sm.set_array([])
plt.colorbar(sm)
plt.show()
           
#%%     
diff = (0, 2) #height difference between path[diff[1]] and path[diff[0]]
m = 1 #number of splitting lines parallel to the long side of the bed
n = 100 #number of splitting lines perpendicular to the long side of the bed
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
        n = 50
        for i in range(1, n + 1):
            point_to_add = [(i*longest_line[0][0] + (n + 1 -i) * longest_line[1][0])*(1/(n+1)), (i*longest_line[0][1] + (n + 1 -i) * longest_line[1][1])*(1/(n+1))]
            ret.append(point_to_add)
        
        slope = np.arctan((longest_line[1][1] - longest_line[0][1])/(longest_line[1][0] - longest_line[0][0])) + np.pi/2
        ints = [LineString([(p[0]- 0.01*np.cos(slope),p[1] - 0.01*np.sin(slope)), (p[0] + 0.01*np.cos(slope),p[1] + 0.01*np.sin(slope))]).intersection(Polygon(points).exterior) for p in ret]
        midpoints = [[[(i*ps[0].x + (m + 1 -i) * ps[1].x)*(1/(m+1)), (i*ps[0].y + (m + 1 -i) * ps[1].y)*(1/(m+1))] for ps in [ints[int(n/3)], ints[int(2*n/3)]]] for i in range(1, m + 1)]
        polys = []
        slopes = [np.arctan((mps[1][1] - mps[0][1])/(mps[1][0] - mps[0][0])) for mps in midpoints]
        lines1 = [LineString([(midpoints[i][0][0]- 0.01*np.cos(slope2), midpoints[i][0][1] - 0.01*np.sin(slope2)), (midpoints[i][0][0] + 0.01*np.cos(slope2), midpoints[i][0][1] + 0.01*np.sin(slope2))]) for i, slope2 in enumerate(slopes)]
        lines2 = [LineString([(p[0]- 0.01*np.cos(slope),p[1] - 0.01*np.sin(slope)), (p[0] + 0.01*np.cos(slope),p[1] + 0.01*np.sin(slope))]) for p in ret]
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

for i, poly in enumerate(allpolys):
    if poly_values[i]:
        plt.fill(*poly.exterior.xy, c=cm.get_cmap(colormap)((np.mean(poly_values[i]) - min_mean)/(max_mean-min_mean)))
        
sm = cm.ScalarMappable(norm=Normalize(min_mean, max_mean), cmap=colormap)
sm.set_array([])
plt.colorbar(sm)
plt.show()

#%%
schema = {
    'geometry': 'Polygon',
    'properties': {'mean_height_diff': 'float'},
}

with fiona.open(wd + '/height_diff_polygons_0724-0804.shp', 'w', crs={'init': 'epsg:4326'}, driver='ESRI Shapefile', schema=schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    for i, p in enumerate(allpolys):
        c.write({
            'geometry': mapping(p),
            'properties': {'mean_height_diff': np.mean(poly_values[i])},
        })