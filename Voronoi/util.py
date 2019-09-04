from shapely.geometry import Point, Polygon, LinearRing
import numpy as np
import fiona
from tqdm import trange
import scipy.stats as st
from pyqtree import Index
from collections import OrderedDict

def get_areas_and_lengths(vor, convex_hull):
    a = []
    lengths = []
    for i in range(len(vor.regions)):
        if -1 not in vor.regions[i] and vor.regions[i] and len(vor.regions[i]) > 2:
            vs = []
            on_border = False
            for j in vor.regions[i]:
                vs.append((vor.vertices[j][0], vor.vertices[j][1]))
                if not Point(vor.vertices[j][0], vor.vertices[j][1]).within(convex_hull):
                    on_border = True
                
            if vs:
                if not on_border:
                    poly = Polygon(vs)
                    a.append(poly.area)
                    lengths.append(poly.length)
            
    return a, lengths

def get_slope_and_dist(p, q):
    if p[0] == q[0] and p[1] == q[1]:
        return 0, 0
    [p, q] = sorted([p, q], key=lambda v : v[0])
    return np.arctan((p[1] - q[1])/ (p[0] - q[0])), np.sqrt((p[1] - q[1])**2 + (p[0] - q[0])**2)

def find_points_in_line(ps, ci, vor):
    lines = []
    lines.append([ps[0]])
    for i in range(1, len(ps)):
        p = vor.points[(np.where(vor.point_region == ps[i].id))[0][0]]
        added = False
        for j in range(len(lines)):
            inline = []
            for poly in lines[j]:
                q = vor.points[(np.where(vor.point_region == poly.id))[0][0]]
                slope, dist = get_slope_and_dist(p, q)
                if slope > ci[0] and slope < ci[1]:
                    inline.append(True)
                else:
                    inline.append(False)
                    
            if sum(inline) == len(lines[j]):
                lines[j].append(ps[i])
                added = True
                
        if not added:
            lines.append([ps[i]])
            
    points = []
    for i in range(len(lines)):
        line = []
        for j in range(len(lines[i])):
            line.append(vor.points[(np.where(vor.point_region == lines[i][j].id))[0][0]])
        line = sorted(line, key = lambda p : p[0])
        points.append(line)
    return points

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
    
def get_convex_hull(plants):
    poly = Polygon(zip(plants[:,0], plants[:,1]))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    return polygon

def get_confidence_interval_areas(a):
    iqr = np.abs(np.percentile(a, 75) - np.percentile(a, 25))
    return (np.percentile(a,25) - 1.5 * iqr, np.percentile(a, 75) + 1.5 * iqr)

def get_confidence_interval_slopes(slopes, conf):
    return st.t.interval(conf, len(slopes) - 1, loc=np.median(slopes), scale = np.std(slopes))

def readable_values(plants):
    f = 10000
    mean_x_coord = np.mean(plants[:,0])
    mean_y_coord = np.mean(plants[:,1])
    plants_i = np.zeros(plants.shape)
    plants_i[:,0] = f*(plants[:,0] - mean_x_coord)
    plants_i[:,1] = f*(plants[:,1] - mean_y_coord)
    return plants_i, mean_x_coord, mean_y_coord

def readable_values_inv(plants, mean_x_coord, mean_y_coord):
    f = 10000
    plants_i = np.zeros(plants.shape)
    plants_i[:,0] = plants[:,0] / f + mean_x_coord
    plants_i[:,1] = plants[:,1] / f + mean_y_coord
    return plants_i

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