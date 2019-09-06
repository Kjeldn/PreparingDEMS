from shapely.geometry import Point, Polygon, LinearRing, LineString
import numpy as np
import fiona
from tqdm import trange
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

#%%
def ci_slopes(p, q, slope_mean, delta):
    _, dist = get_slope_and_dist(p, q)
    return (slope_mean - np.arctan(delta/dist), slope_mean + np.arctan(delta/dist))

def find_points_in_line2(ps, ci, vor):
    ci_mean = (ci[1] + ci[0])/2
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
                    if abs(ci_mean - slope) < abs(ci_mean - closest_slope):
                        closest_slope_p = q
                        closest_slope = slope
                        
        ci_s = ci_slopes(coords[points.index(p)], coords[points.index(closest_slope_p)], ci_mean, 0.01)
        
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
                    if all(all(get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])[0] > ci_slopes(coords[points.index(p)], coords[points.index(q)], ci_mean, 0.05)[0] and get_slope_and_dist(coords[points.index(p)], coords[points.index(q)])[0] < ci_slopes(coords[points.index(p)], coords[points.index(q)], ci_mean, 0.05)[1] for q in lines[k]) for p in lines[j]):                            
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

# =============================================================================
# amr = list(adjacent_missed_regions[63])
# lines = find_points_in_line2(amr, ci_s, vor)
# for line in lines:
#     if len(line) > 1:
#         line  = np.array(line)
#         plt.plot(line[:,0], line[:,1])
#     
# points = np.array([vor.points[(np.where(vor.point_region == vp.id))[0][0]] for vp in amr])
# plt.scatter(points[:,0], points[:,1])
# =============================================================================

#%%
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

def get_confidence_interval(a):
    iqr = np.abs(np.percentile(a, 75) - np.percentile(a, 25))
    return (np.percentile(a,25) - 1.5 * iqr, np.percentile(a, 75) + 1.5 * iqr)

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
    try:
        plants_i[:,0] = plants[:,0] / f + mean_x_coord
        plants_i[:,1] = plants[:,1] / f + mean_y_coord
        return plants_i
    except:
        return np.array([])

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