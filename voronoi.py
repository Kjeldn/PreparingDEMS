import fiona
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import OrderedDict
from pyqtree import Index

path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\20190717_count.shp"
dst = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\20190717_count_missed.shp"

class Voronoi_polygon:
    def __init__(self, id, vi, vc):
        self.id = id
        self.vi = vi
        self.vc = vc
        
    def __eq__(self, other):
        if self.id != other.id:
            return False
        if self.vi != other.vi:
            return False
        return True
    
    def __hash__(self):
        return self.id * sum(self.vi)
    
    def is_adjacent(self, other):
        for i in self.vi:
            if i in other.vi:
                return True
        return False
    
    def area(self):
        return Polygon(self.vc).area
    
    def adjacent_line(self, other):
        vs = []
        for i in range(len(self.vi)):
            if self.vi[i] in other.vi:
                vs.append(self.vc[i])
        return vs
    
    def length(self):
        return Polygon(self.vc).length
    
    def compactness(self):
        return (self.area() * 4 * np.pi) / self.length()
        
        
def get_convex_hull(plants):
    poly = Polygon(zip(plants[:,0], plants[:,1]))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    return polygon

def get_confidence_interval(a, conf):
    return st.t.interval(conf, len(a) - 1, loc=np.mean(a), scale = np.std(a))

def find_adjacent_polygons(ps):
    aps = []
    for p in ps:
        for q in ps:
            if p != q and p.is_adjacent(q): 
                aps.append({p, q})
                
    for i in range(len(aps)):
        found = False
        for j in range(len(aps)):
            for k in range(len(aps)):
                for r in aps[j]:
                    if r in aps[k] and k != j:
                        aps[k] = aps[k].union(aps[j])
                        del aps[j]
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    return aps

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
                slope = (p[1] - q[1])/ (p[0] - q[0])
                if (ci[0] < 0 and slope > ci[0] and slope < ci[1]) or (ci[0] > 0 and slope > ci[0] and slope < ci[1]):
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

def get_slope_and_dist(p, q):
    if p[0] == q[0] and p[1] == q[1]:
        return 0, 0
    return (p[1] - q[1])/ (p[0] - q[0]), np.sqrt((p[1] - q[1])**2 + (p[0] - q[0])**2)

def get_point_in_polygon(poly, vor):
    return vor.points[(np.where(vor.point_region == poly.id))[0][0]]

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

def get_areas_and_lengths(vor, convex_hull):
    a = []
    lengths = []
    for i in range(len(vor.regions)):
        if -1 not in vor.regions[i] and vor.regions[i] and len(vor.regions[i]) > 2:
            vs = []
            for j in vor.regions[i]:
                if Point(vor.vertices[j][0], vor.vertices[j][1]).within(convex_hull):
                    vs.append((vor.vertices[j][0], vor.vertices[j][1]))
                else:
                    vs = []
                    break
                
            if vs:
                poly = Polygon(vs)
                a.append(poly.area)
                lengths.append(poly.length)


    for i in range(len(a) -1, -1, -1):
        if abs(lengths[i] - np.median(lengths)) > np.median(lengths):
            del a[i]
            
    return a, lengths

def get_large_and_small_regions(vor, convex_hull, ci, lengths):
    missed_regions = []
    small_regions = []
    for i in range(len(vor.regions)):
        if not -1 in vor.regions[i] and vor.regions[i]:
            vs = []
            for j in vor.regions[i]:
                if Point(vor.vertices[j][0], vor.vertices[j][1]).within(convex_hull):
                    vs.append((vor.vertices[j][0], vor.vertices[j][1]))
                else:
                    vs = []
                    break
            
            if vs:
                poly = Polygon(vs)
                if poly.area > ci[1]: # and abs(poly.length - np.median(lengths)) < 0.5 * np.median(lengths):
                    missed_regions.append(Voronoi_polygon(i, vor.regions[i], [list(vor.vertices[k]) for k in vor.regions[i]]))
                elif poly.area < ci[0]:
                    small_regions.append(Voronoi_polygon(i, vor.regions[i], [list(vor.vertices[k]) for k in vor.regions[i]]))
    return missed_regions, small_regions

def find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions):
    missed_points = []
    for s in adjacent_missed_regions:
        if len(s) == 2:
            aps = list(s)
            line = aps[0].adjacent_line(aps[1])
            if line and len(line) > 1:
                missed_points.append([(line[0][0] + line[1][0])/ 2, (line[0][1] + line[1][1])/2])
    return missed_points

def get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions):
    dists = []
    slopes = []
    for amr in adjacent_missed_regions:
        if len(amr) == 2:
            it = iter(amr)
            p1 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            p2 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            slope, dist = get_slope_and_dist(p1, p2)
            slopes.append(slope)
            dists.append(dist)
            
    mean_s= np.mean(slopes)
    for i in range(len(slopes)):
        if abs(np.arctan(slopes[i]) - np.arctan(mean_s)) > np.pi / 4:
            slopes[i] = mean_s
    return slopes, dists

def find_missed_points_in_regions(adjacent_missed_regions, vor, ci_s, dists, spindex):
    missed_points = []
    for i in range(len(adjacent_missed_regions)):
        if len(adjacent_missed_regions[i]) != 2:
            l = list(adjacent_missed_regions[i])
            lines = find_points_in_line(l, ci_s, vor)
            for line in lines:
                for c in range(len(line) - 1):
                    dist = np.sqrt((line[c][1] - line[c + 1][1])**2 + (line[c][0] - line[c + 1][0])**2)
                    n_p = int(dist/(np.mean(dists)/2) + 0.5) - 1
                    if n_p > 0:# and n_p < 8:
                        mps = fill_points_in_line(line[c], line[c + 1], n_p, spindex, np.mean(dists)/4)
                        for mp in mps:
                            if mp not in missed_points:
                                missed_points.append(mp)
    return missed_points

def plot_voronoi_diagram(vor, plot_points, missed_regions, small_regions):
    voronoi_plot_2d(vor, show_vertices=False)
    if len(plot_points) > 0:
        plt.scatter(plot_points[:,0], plot_points[:,1], color='k')
    for r in missed_regions:
        plt.fill(*zip(*r.vc), color="r", alpha = 0.5)
        
    for r in small_regions:
        plt.fill(*zip(*r.vc), color="g", alpha = 1)
        
    plt.show()
    
def get_missing_points(plants, plot=False):
    convex_hull = get_convex_hull(np.array(plants))
    vor = Voronoi(plants)
    a, lengths = get_areas_and_lengths(vor, convex_hull)                     
    ci = get_confidence_interval(a, 0.95)
    missed_regions, small_regions = get_large_and_small_regions(vor, convex_hull, ci, lengths)         
    adjacent_missed_regions = find_adjacent_polygons(missed_regions)                    
    missed_points = find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions)          
    slopes, dists = get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    ci_s = get_confidence_interval(slopes, 0.95)
    missed_points = missed_points + find_missed_points_in_regions(adjacent_missed_regions, vor, ci_s, dists, spindex)
    if plot:
        plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points


#%% plants
with fiona.open(path) as src:
    plants = []
    src_driver = src.driver
    src_crs = src.crs
    src_schema = src.schema
    for i in range(len(src)):
        if src[i]['geometry']:
            plants.append([src[i]['geometry']['coordinates'][0][0], src[i]['geometry']['coordinates'][0][1]])

#plants = sorted(sorted(plants, key=lambda a : a[0]), key = lambda a: a[1]) ## use if point data is not ordered
plants = np.array(plants)
for i in range(plants.shape[0]-1, -1, -1):
    if (plants[i][0] == 0 and plants[i][1] == 0):
        plants = np.delete(plants, i, 0)

missed_points_coord = []
n = 5000
overlap = 1000
for i in range(int(np.ceil(len(plants)/n))):
#for i in range(4,5):
    print(i / int(len(plants)/n) * 100, '%')
    offset = n if (i + 1) * n < len(plants) else len(plants) - i * n
    offset = offset + overlap if i * n + offset + overlap < len(plants) else offset
    plants_i, mean_x_coord, mean_y_coord = readable_values(plants[i * n: i * n + offset, :])
    spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
    for plant in plants_i:
        spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
    
    if abs(get_confidence_interval(plants_i[:,0], 0.95)[0]) + abs(get_confidence_interval(plants_i[:,0], 0.95)[1]) > 10:
        plants2 = []
        plants1 = []
        for i in range(plants_i.shape[0]):
            if plants_i[i, 0] > 0:
                plants1.append(list(plants_i[i, :]))
            else:
                plants2.append(list(plants_i[i, :]))
        plants_list = [plants1, plants2]    
    
        missed_points = []
        for j in range(2):
            missed_points = missed_points + get_missing_points(plants_list[j])
                                            
        missed_points_coord = missed_points_coord + list(readable_values_inv(np.array(missed_points), mean_x_coord, mean_y_coord))                
        
    else:
        missed_points_coord = missed_points_coord + list(readable_values_inv(np.array(get_missing_points(plants_i)), mean_x_coord, mean_y_coord))
        
missed_points_coord = np.array(missed_points_coord)
missed_points_qtree = Index(bbox=(np.amin(missed_points_coord[:, 0]), np.amin(missed_points_coord[:,1]), np.amax(missed_points_coord[:,0]), np.amax(missed_points_coord[:, 1])))
d_y = 1/444444.0 ## 0.25 meters
d_x = np.cos(missed_points_coord[0][1] / (180 / np.pi))/444444.0 ## 0.25 meters
for mp in missed_points_coord:
    if not missed_points_qtree.intersect((mp[0] - d_x, mp[1] - d_y, mp[0] + d_x, mp[1] + d_y)):
        missed_points_qtree.insert(mp, (mp[0] - d_x, mp[1] - d_y, mp[0] + d_x, mp[1] + d_y))

missed_points_coord = missed_points_qtree.intersect(bbox=(np.amin(missed_points_coord[:, 0]), np.amin(missed_points_coord[:,1]), np.amax(missed_points_coord[:,0]), np.amax(missed_points_coord[:, 1])))

with fiona.open(dst, 'w', crs = src_crs, driver =src_driver, schema ={'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])}) as dst:
    for i in range(len(missed_points_coord)):
        dst.write({
                'geometry': {
                        'type': 'Point',
                        'coordinates': (missed_points_coord[i][0], missed_points_coord[i][1])
                        },
                'properties': OrderedDict([('name', 'missed point')])
    })
    

    