from shapely.geometry import Point, Polygon, LinearRing
import numpy as np
import fiona
from tqdm import trange
from pyqtree import Index
from collections import OrderedDict

"""
Get the areas and lengths of all Voronoi_polygons inside convex_hull.

Parameters
-----------
vor : Voronoi_diagram
    the Voronoi diagram which describes all Voronoi_polygons and points
convex_hull : shapely Polygon
    Convex hull of Voronoi points to determine if Voronoi polygon is on border, if on border it is ignored

Returns
-----------
areas : list of float
    Areas of all Voronoi_polygons in the Voronoi diagram
lengths : list of float
    Circumference of all Voronoi_polygons in the Voronoi diagram
"""
def get_areas_and_lengths(vor, convex_hull):
    areas = []
    lengths = []
    for i in range(len(vor.regions)):
        if -1 not in vor.regions[i] and vor.regions[i] and len(vor.regions[i]) > 2 and all(Point(vor.vertices[j]).within(convex_hull) for j in vor.regions[i]):
            poly = Polygon([(vor.vertices[j][0], vor.vertices[j][1]) for j in vor.regions[i]])
            areas.append(poly.area)
            lengths.append(poly.length)
            
    return areas, lengths

"""
Get slope (in radians) and dist between two points.

Parameters
-----------
p,q : coordinates
    The two points for which the slope and dist is determined
    
Returns
-----------
slope : float
    The slope between p and q
dist : float
    The distance between p and q
"""
def get_slope_and_dist(p, q):
    if p[0] == q[0] and p[1] == q[1]:
        return 0, 0
    [p, q] = sorted([p, q], key=lambda v : v[0])
    return np.arctan((p[1] - q[1])/ (p[0] - q[0])), np.sqrt((p[1] - q[1])**2 + (p[0] - q[0])**2)

"""
Get the confidence interval for the slope. This interval is used to determine if Voronoi points are in the same line.
The interval is dependent on the distance between points p and q. When a straight triangle is drawn from the line between
p and q and a line with slope slope_mean from p the third line is maximum delta.

     .p
    /|
   / |
  .--.q
   delta
   
Parameters
-----------
p, q : coordinates
    The two points for which the confidence interval is computed
slope_field : float
    The slope of the field
delta : float
    The length of opposite side of p in a straight triangle

Returns
-----------
confidence_interval : tuple(len=2) of float
    Confidence interval for the slope
"""
def ci_slopes(p, q, slope_field, delta):
    _, dist = get_slope_and_dist(p, q)
    return (slope_field - np.arctan(delta/dist), slope_field + np.arctan(delta/dist))

"""
Group all Voronoi points in a set of Voronoi_polygons which are in the same line dependent on
the slope of the field.
   
Parameters
-----------
ps : set of Voronoi_polygons
    All Voronoi_polygons for which the points inside have to be grouped
slope_field : float
    The slope of the field
vor : Voronoi_diagram
    The Voronoi diagram describing it all

Returns
-----------
lines_coord : list of list of coordinates
    List of all Voronoi_points in the same line in the set of Voronoi_polygons grouped together
"""
def find_points_in_line(ps, slope_field, vor):
    coords = [vor.points[np.where(vor.point_region == p.id)[0][0]] for p in ps]
    points = [np.where(vor.point_region == p.id)[0][0] for p in ps]
    lines = []
    for p in points:
        closest_slope_p = None
        closest_slope = 0
        for q in points:
            if p != q:
                if not closest_slope_p:
                    closest_slope_p = q
                    closest_slope, _ = get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])
                else:
                    slope, _ = get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])
                    if abs(slope_field - slope) < abs(slope_field - closest_slope):
                        closest_slope_p = q
                        closest_slope = slope
                        
        ci_s = ci_slopes(coords[points.index(p)], coords[points.index(closest_slope_p)], slope_field, 0.02)
        
        if closest_slope > ci_s[0] and closest_slope < ci_s[1]:
            lines.append([p, closest_slope_p])
        
    for i in range(len(lines)):
        found = False
        for j in range(len(lines)):
            for k in range(len(lines)):
                if k != j and any(p in lines[j] for p in lines[k]):
                    newline = list(set(lines[j] + lines[k]))
                    if j < k:
                        del lines[k]
                        del lines[j]
                    else:
                        del lines[j]
                        del lines[k]
                    lines.append(newline)
                    found = True
                    break
            if found:
                break
            
    for i in range(len(lines)):
        found = False
        for j in range(len(lines)):
            for k in range(len(lines)):
                if j !=k:
                    if all(all(get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])[0] > ci_slopes(coords[points.index(p)], coords[points.index(q)], slope_field, 0.05)[0] and get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])[0] < ci_slopes(coords[points.index(p)], coords[points.index(q)], slope_field, 0.05)[1] for q in lines[k]) for p in lines[j]):                            
                        newline = list(set(lines[j] + lines[k]))
                        if j < k:
                            del lines[k]
                            del lines[j]
                        else:
                            del lines[j]
                            del lines[k]
                        lines.append(newline)
                        found = True
                        break
            if found:
                break
    
    lines_coord = []
    for line in lines:
        lines_coord.append(sorted([coords[points.index(p)] for p in line], key=lambda a: a[0]))
    return lines_coord

"""
Returns n equally spaced points in the line between p and q.
If one of these points has distance smaller thatn d to a Voronoi point an empty list is returned.
   
Parameters
-----------
p,q : coordinates
    The points where in between points are filled
n : int
    The number of points which have to be filled in
spindex : pyqtree Index
    Quad tree to determine if one the missed points is too close to already existing points
d : float
    The max distance the returned points can have to already existing points

Returns
-----------
ret : list(len=n) of coordinates
    The points filled in the line between p and q
"""
def fill_points_in_line(p, q, n, spindex, d):
    ret = []
    is_on_top_of_point = []
    for i in range(1, n + 1):
        point_to_add = [(i*p[0] + (n + 1 -i) * q[0])*(1/(n+1)), (i*p[1] + (n + 1 -i) * q[1])*(1/(n+1))]
        ret.append(point_to_add)
        if len(spindex.intersect((point_to_add[0] - d, point_to_add[1] - d, point_to_add[0] + d, point_to_add[1] + d))) == 0:
            is_on_top_of_point.append(False)
        else:
            is_on_top_of_point.append(True)
    if sum(is_on_top_of_point) == 0:
        return ret
    else:
        return []
 
"""
Gets a convex hull around points.
   
Parameters
-----------
plants : numpy array of coordinates
    The points around which a convex hull is given

Returns
-----------
polygon : shapely Polygon
    The convex hull around plants
"""
def get_convex_hull(plants):
    poly = Polygon(zip(plants[:,0], plants[:,1]))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    return polygon

"""
Get confidence interval of list of values determined by the whiskers in a boxplot.
   
Parameters
-----------
a : list of float
    List of values for which the confidence interval is determined

Returns
-----------
confidence_interval : tuple(len=2) of float
    Whiskers based on values of a.
"""
def get_confidence_interval(a):
    iqr = np.abs(np.percentile(a, 75) - np.percentile(a, 25))
    return (np.percentile(a, 25) - 1.5 * iqr, np.percentile(a, 75) + 1.5 * iqr)

"""
Scipy Voronoi could not handle the original coordinates, this transforms the values to more readable values.
   
Parameters
-----------
plants : numpy array of coordinates
    List of coordinates

Returns
-----------
plants_i : numy array of coordinates
    List of more readable coordinates
mean_x_coord : float
    The mean of x of the original coordinates, used for inverse of this function
mean_y_coord : float
    The mean of y of the original coordinates, used for inverse of this function
"""
def readable_values(plants):
    f = 10000
    mean_x_coord = np.mean(plants[:,0])
    mean_y_coord = np.mean(plants[:,1])
    plants_i = np.zeros(plants.shape)
    plants_i[:,0] = f*(plants[:,0] - mean_x_coord)
    plants_i[:,1] = f*(plants[:,1] - mean_y_coord)
    return plants_i, mean_x_coord, mean_y_coord

"""
Inverse of readable_values. Returns the coordinates of the readable variant of points.
   
Parameters
-----------
plants : numpy array of coordinates
    List of coordinates
mean_x_coord : float
    The mean of x of the original coordinates
mean_y_coord : float
    The mean of y of the original coordinates

Returns
-----------
plants_i : numy array of coordinates
    List of coordinates
"""
def readable_values_inv(plants, mean_x_coord, mean_y_coord):
    f = 10000
    plants_i = np.zeros(plants.shape)
    try:
        plants_i[:,0] = plants[:,0] / f + mean_x_coord
        plants_i[:,1] = plants[:,1] / f + mean_y_coord
        return plants_i
    except:
        return np.array([])

"""
Get the points in a shape file.
   
Parameters
-----------
path : string
    Path of the shape file to open.

Returns
-----------
plants : list of coordinates
    The coordinates of the points in the shape file
src_driver : ESRI Shapefile
    The driver of the shapefile, used for writing the shapefile of missed points
src_crs : string
    The CRS of the shapefile, used for writing the shapefile of missed points
src_schema : OrderedDict
    The schema of the shapefile, used for writing the shapefile of missed points
"""
def open_shape_file(path):
    with fiona.open(path) as src:
        plants = []
        src_driver = src.driver
        src_crs = src.crs
        src_schema = src.schema
        for i in trange(len(src), desc='opening plants'):
            if src[i]['geometry']:
                if src[i]['geometry']['type'] == 'MultiPoint':
                    plants.append([src[i]['geometry']['coordinates'][0][0], src[i]['geometry']['coordinates'][0][1]])
                elif src[i]['geometry']['type'] == 'Point':
                    plants.append([src[i]['geometry']['coordinates'][0], src[i]['geometry']['coordinates'][1]])
    return plants, src_driver, src_crs, src_schema

"""
Write a new shapefile.
   
Parameters
-----------
path : string
    Path of the shape file to write.
missed_points_coord : list of coordinates
    List of points to write in new shapefile
crs : string
    The CRS of the new shapefile
driver : ESRI Shapefile
    The driver of the new shapefile
schema : OrderedDict
    The schema of the new shapefile
"""
def write_shape_file(path, missed_points_coord, crs, driver, schema):
    with fiona.open(path, 'w', crs = crs, driver =driver, schema =schema) as dst:
        for i in trange(len(missed_points_coord), desc='writing new shapefile'):
            dst.write({
                    'geometry': {
                            'type': 'Point',
                            'coordinates': (missed_points_coord[i][0], missed_points_coord[i][1])
                            },
                    'properties': OrderedDict([('name', 'missed point')])
                    })

"""
Returned a trimmed list of missed points where missed points which have a distance smaller that 0.25 meters are removed.
   
Parameters
-----------
missed_points_coord : list of coordinates
    Missed points which have to be checked for overlapping points
    
Returns
-----------
missed_points_coord : list of coordinates
    Missed points where overlapping points are removed
"""
def remove_overlapping_points(missed_points_coord):
    missed_points_qtree = Index(bbox=(np.amin(missed_points_coord[:, 0]), np.amin(missed_points_coord[:,1]), np.amax(missed_points_coord[:,0]), np.amax(missed_points_coord[:, 1])))
    d_y = 1/444444.0 ## 0.25 meters
    d_x = np.cos(missed_points_coord[0][1] / (180 / np.pi))/444444.0 ## 0.25 meters
    for i in trange(len(missed_points_coord), desc='check for double points'):
        mp = missed_points_coord[i]
        if not missed_points_qtree.intersect((mp[0] - d_x, mp[1] - d_y, mp[0] + d_x, mp[1] + d_y)):
            missed_points_qtree.insert(mp, (mp[0] - d_x, mp[1] - d_y, mp[0] + d_x, mp[1] + d_y))
    
    missed_points_coord = missed_points_qtree.intersect(bbox=(np.amin(missed_points_coord[:, 0]), np.amin(missed_points_coord[:,1]), np.amax(missed_points_coord[:,0]), np.amax(missed_points_coord[:, 1])))
    return missed_points_coord