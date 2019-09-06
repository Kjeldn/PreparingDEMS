from shapely.geometry import Polygon, Point, LineString
import numpy as np
import util
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib.pyplot as plt
from tqdm import trange

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
    
class Voronoi_diagram:
    def __init__ (self, points):
        self.vor = Voronoi(points)
        self.vertices = self.vor.vertices
        self.points = self.vor.points
        self.ridge_points = self.vor.ridge_points
        self.ridge_vertices = self.vor.ridge_vertices
        self.regions = self.vor.regions
        self.point_region = self.vor.point_region
        
    def coords_p(self, i):
        return self.points[i]
    
    def coords_v(self, i):
        return self.vertices[i]
    
    def coords_re(self, i):
        return [self.coords_v(j) for j in self.regions[i]] if -1 not in self.regions[i] else []
    
    def coord_ri(self, i):
        return [self.coords_v(j) for j in self.ridge_vertices[i]] if -1 not in self.ridge_vertices[i] else []
    
    def show(self):
        voronoi_plot_2d(self)
        
    def remove_ridges(self, convex_hull):
        for i in range(len(self.ridge_vertices) -1, -1, -1):
            if all(not Point(v).inside(convex_hull) for v in self.coord_ri(i)):
                del self.ridge_vertices[i]
    
    def clip_ridges(self, convex_hull):
        clipped_ridges = []
        for i in range(len(self.ridge_points)):
            [v1, v2] = self.ridge_vertices[i]
            if -1 not in [v1, v2] and Point(self.vertices[v1]).within(convex_hull) != Point(self.vertices[v2]).within(convex_hull):
                x = convex_hull.exterior.intersection(LineString([self.vertices[v1], self.vertices[v2]]))
                self.vertices = np.append(self.vertices, [[x.x, x.y]], axis=0)
                if Point(self.vertices[v1]).within(convex_hull):
                    self.ridge_vertices[i] = [v1, len(self.vertices) - 1]
                else:
                    self.ridge_vertices[i] = [len(self.vertices) - 1, v2]
                clipped_ridges.append({'src': [v1, v2], 'dest': self.ridge_vertices[i]})
        return clipped_ridges
    
    def clip_regions(self, convex_hull):
        clipped_ridges = self.clip_ridges(convex_hull)
        for i in range(len(self.regions)):
            if -1 not in self.regions[i] and not all(Point(self.vertices[v]).within(convex_hull) for v in self.regions[i]):
                vs = self.regions[i]
                vs_i = list(filter(lambda i : not Point(self.vertices[vs[i]]).within(convex_hull), list(np.arange(len(vs)))))
                hole_i = list(filter(lambda i : vs_i[i + 1] - vs_i[i] != 1, np.arange(len(vs_i) - 1)))
                if hole_i:
                    vs_i = vs_i[hole_i[0] + 1:] + vs_i[:hole_i[0] + 1]
                filtered_ridges1 = list(filter(lambda r: vs[(vs_i[0] - 1)% len(vs)] in r['src'] and vs[vs_i[0]] in r['src'], clipped_ridges))
                filtered_ridges2 = list(filter(lambda r: vs[(vs_i[-1] + 1) % len(vs)] in r['src'] and vs[vs_i[-1]] in r[ 'src'], clipped_ridges))
                vs_i = sorted(vs_i)
                for j in range(len(vs_i) -1, -1, -1):
                    del vs[vs_i[j]]
                if filtered_ridges1:
                    dst1 = filtered_ridges1[0]['dest'][0] if filtered_ridges1[0]['dest'][0] != filtered_ridges1[0]['src'][0] else filtered_ridges1[0]['dest'][1]
                    vs.insert(vs_i[0], dst1)
                if filtered_ridges2: 
                    dst2 = filtered_ridges2[0]['dest'][0] if filtered_ridges2[0]['dest'][0] != filtered_ridges2[0]['src'][0] else filtered_ridges2[0]['dest'][1]
                    if filtered_ridges1:
                        vs.insert(vs_i[0] + 1, dst2)
                    else:
                        vs.insert(vs_i[0], dst2)
        

def get_point_in_polygon(poly, vor):
    return vor.points[(np.where(vor.point_region == poly.id))[0][0]]

def get_large_and_small_regions(vor, convex_hull, ci, lengths, clipped):
    missed_regions = []
    small_regions = []
    for i in range(len(vor.regions)):
        if not -1 in vor.regions[i] and vor.regions[i]:
            vs = []
            on_border = False
            for j in vor.regions[i]:
                vs.append((vor.vertices[j][0], vor.vertices[j][1]))
                if not Point(vor.vertices[j][0], vor.vertices[j][1]).within(convex_hull):
                    on_border = True
            
            if vs and len(vs) > 2 and (clipped or not on_border):
                poly = Polygon(vs)
                if poly.area > ci[1]:
                    missed_regions.append(Voronoi_polygon(i, vor.regions[i], [list(vor.vertices[k]) for k in vor.regions[i]]))
                elif poly.area < ci[0]:
                    small_regions.append(Voronoi_polygon(i, vor.regions[i], [list(vor.vertices[k]) for k in vor.regions[i]]))
    return missed_regions, small_regions

def find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, ci_s, dists):
    missed_points = []
    for s in adjacent_missed_regions:
        if len(s) == 2:
            it = iter(s)
            p1 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            p2 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            slope, dist = util.get_slope_and_dist(p1, p2)
            n_p = int(dist/(np.median(dists)/2) + 0.5) - 1
            if slope > ci_s[0] and slope < ci_s[1] and n_p > 0:
                missed_points.append([(p1[0] + p2[0])/ 2, (p1[1] + p2[1])/2])
    return missed_points

def get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions):
    dists = []
    slopes = []
    for amr in adjacent_missed_regions:
        if len(amr) == 2:
            it = iter(amr)
            p1 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            p2 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            slope, dist = util.get_slope_and_dist(p1, p2)
            slopes.append(slope)
            dists.append(dist)
            
    median_s= np.median(slopes)
    for i in range(len(slopes) - 1, -1, -1):
        if abs(slopes[i] - median_s) > np.pi / 4:
            del slopes[i]
    return slopes, dists

def find_missed_points_in_regions(adjacent_missed_regions, vor, ci_s, dists, spindex):
    missed_points = []
    for i in range(len(adjacent_missed_regions)):
        if len(adjacent_missed_regions[i]) != 2:
            l = list(adjacent_missed_regions[i])
            lines = util.find_points_in_line2(l, ci_s, vor)
            for line in lines:
                for c in range(len(line) - 1):
                    dist = np.sqrt((line[c][1] - line[c + 1][1])**2 + (line[c][0] - line[c + 1][0])**2)
                    n_p = int(dist/(np.median(dists)/2) + 0.5) - 1
                    if n_p > 0:
                        mps = util.fill_points_in_line(line[c], line[c + 1], n_p, spindex, np.mean(dists)/4)
                        for mp in mps:
                            if mp not in missed_points:
                                missed_points.append(mp)
    return missed_points

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

def plot_voronoi_diagram(vor, plot_points, missed_regions, small_regions, show_vertices=False):
    voronoi_plot_2d(vor, show_vertices=show_vertices)
    if len(plot_points) > 0:
        plt.scatter(plot_points[:,0], plot_points[:,1], color='k')
    for r in missed_regions:
        plt.fill(*zip(*r.vc), color="r", alpha = 0.5)
        
    for r in small_regions:
        plt.fill(*zip(*r.vc), color="g", alpha = 1)
        
    plt.show()
    
def clip(vor, convex_hull, dist):
    initial_index = len(vor.vertices)
        
    clipped_ridges = []    
    for i in trange(len(vor.ridge_vertices), desc='clip ridges'):
        [v1, v2] = vor.ridge_vertices[i]
        if -1 not in [v1, v2] and Point(vor.vertices[v1]).within(convex_hull) != Point(vor.vertices[v2]).within(convex_hull):
            x = convex_hull.exterior.intersection(LineString([vor.vertices[v1], vor.vertices[v2]]))
            vor.vertices = np.append(vor.vertices, [[x.x, x.y]], axis=0)
            if Point(vor.vertices[v1]).within(convex_hull):
                vor.ridge_vertices[i] = [v1, len(vor.vertices) - 1]
            else:
                vor.ridge_vertices[i] = [len(vor.vertices) - 1, v2]
            clipped_ridges.append({'src': [v1, v2], 'dest': vor.ridge_vertices[i]})
    
    clipped_ridges = np.array(clipped_ridges)
    for i in trange(len(vor.regions) -1, -1, -1, desc='amend regions'):
        vs = vor.regions[i]
        if -1 not in vs and not all(Point(vor.vertices[v]).within(convex_hull) for v in vs):
            for j in range(len(vs)):
                for r in clipped_ridges:
                    if r['src'] == [vs[j], vs[(j + 1) % len(vs)]] or r['src'] == [vs[(j + 1) % len(vs)], vs[j]]:
                        if r['src'][0] == r['dest'][0]:
                            src = r['src'][1]
                            dst = r['dest'][1]
                        else:
                            src = r['src'][0]
                            dst = r['dest'][0]
                            
                        if vs[j] == src:
                            vs[j] = dst
                        else:
                            vs[(j + 1) % len(vs)] = dst
                        vor.ridge_vertices = np.append(vor.ridge_vertices, [[vs[j], vs[(j + 1) % len(vs)]]], axis=0)
                        break
            for j in range(len(vs) -1, -1, -1):
                if vs[j] < initial_index and not Point(vor.vertices[vs[j]]).within(convex_hull):
                        del vs[j]
            vor.regions[i] = vs
        else:
            del vor.regions[i]
                
    return vor