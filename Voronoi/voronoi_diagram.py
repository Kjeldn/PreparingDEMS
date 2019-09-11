from shapely.geometry import Polygon, Point, LineString
import numpy as np
import util
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib.pyplot as plt

"""
Extending of Voronoi regions found in the Voronoi diagram,
Needs custom hash code functions to be used in sets and equals function to compare them.
"""
class Voronoi_polygon:
    def __init__(self, index, vi, vc):
        self.id = index
        self.vi = vi
        self.vc = vc
        
    def __key(self):
        return (self.id, len(self.vi))
        
    def __eq__(self, other):
        if self.id != other.id:
            return False
        if self.vi != other.vi:
            return False
        return True
    
    def __hash__(self):
        return hash(self.__key())
    
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

"""
Extension of the Voronoi diagram given by the scipy package,
Most important feature: clip_regions
"""  
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
    
    """
    Clip Voronoi ridges which intersects the polygon given and add new Voronoi vertices
    for intersections between ridges and the exterior of the polygon
    
    Parameters
    -----------
    polygon : Polygon
        a shapely Polygon for which the ridges are clipped
    
    Returns
    -----------
    clipped regions : list[{ 'src':[], 'dest':[] }]
        a list of objects which describes the indices of the clipped ridge before clipping
        and the indices of the clipped ridge after clipping
    """
    def clip_ridges(self, polygon):
        clipped_ridges = []
        for i in range(len(self.ridge_points)):
            [v1, v2] = self.ridge_vertices[i]
            if -1 not in [v1, v2] and Point(self.vertices[v1]).within(polygon) != Point(self.vertices[v2]).within(polygon):
                x = polygon.exterior.intersection(LineString([self.vertices[v1], self.vertices[v2]]))
                self.vertices = np.append(self.vertices, [[x.x, x.y]], axis=0)
                if Point(self.vertices[v1]).within(polygon):
                    self.ridge_vertices[i] = [v1, len(self.vertices) - 1]
                else:
                    self.ridge_vertices[i] = [len(self.vertices) - 1, v2]
                clipped_ridges.append({'src': [v1, v2], 'dest': self.ridge_vertices[i]})
        return clipped_ridges
    
    """
    Clip Voronoi regions of self which intersects the polygon given
    
    Parameters
    -----------
    polygon : Polygon
        a shapely Polygon for which the ridges are clipped
    
    """
    def clip_regions(self, polygon):
        clipped_ridges = self.clip_ridges(polygon)
        for i in range(len(self.regions)):
            for j in range(len(self.regions[i])-1, -1, -1):
                if self.regions[i][j] == -1:
                    del self.regions[i][j]
            if not all(Point(self.vertices[v]).within(polygon) for v in self.regions[i]):
                vs = self.regions[i]
                vs_i = list(filter(lambda i : not Point(self.vertices[vs[i]]).within(polygon), list(np.arange(len(vs)))))
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
                
        
"""
Get the coordinates of the point inside a Voronoi_polygon

Parameters
-----------
poly : Voronoi_polygon
    the polygon for which the point inside is given
vor : Voronoi_diagram
    the Voronoi diagram which describes all Voronoi_polygons and points

Returns
-----------
coordinates : list(len=2) of coordinates
    coordinates of the point inside poly
"""
def get_point_in_polygon(poly, vor):
    return vor.points[(np.where(vor.point_region == poly.id))[0][0]]

"""
Get all large and small regions in the Voronoi diagram

Parameters
-----------
vor : Voronoi_diagram
    the Voronoi diagram for which the large and small regions are searched
convex_hull : shapely Polygon
    convex_hull of points in Voronoi diagram to determine if a Voronoi region is on border
ci : tuple(len = 2) 
    confidence interval of areas of regions
clipped : Boolean
    whether the voronoi_diagram is clipped, if not regions on the border are ignored
    
Returns
-----------
missed_regions : list of Voronoi_polygon
    all Voronoi_polygons which have an area larger than the confidence interval
small_regions : list of Voronoi_polygon
    all Voronoi_polygons which have an area smaller than the confidence interval
"""
def get_large_and_small_regions(vor, convex_hull, ci, clipped):
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

"""
Determine missed points in between two Voronoi points which have Voronoi_polygons which are adjacent and flagged as large
(and no other adjacent polygons are flagged as large). The number of missed points is determined by the median of distances between such two Voronoi points
in the first iteration and by the mean_dist in further iterations.
If the slope of the two Voronoi points is too different from the slope_field than they are ignores

Parameters
-----------
adjacent_missed_regions : list of set of Voronoi_polygon
    Contains a set of Voronoi_polygons which are flagged as large and all are adjacent
vor : Voronoi_diagram
    The Voronoi diagram containing it all
slope_field : float
    The slope of the field in radians
dists : list of float
    List of distances between two Voronoi points for which the Voronoi_polygons are adjacent
first_it : Boolean
    True if this function is triggered in the first iteration
mean_dist : float
    The mean distance of all pairs of Voronoi points in large Voronoi regions from the first iteration, only used if first_it is True

Returns
-----------
missed_point : list of coordinates
    coordinates of all missed points in adjacent_missed_regions which have length 2
"""
def find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, slope_field, dists, first_it=True, mean_dist=None):
    missed_points = []
    for s in adjacent_missed_regions:
        if len(s) == 2:
            it = iter(s)
            p1 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            p2 = vor.points[(np.where(vor.point_region == next(it).id))[0][0]]
            slope, dist = util.get_slope_and_dist(p1, p2)

            if first_it:
                n_p = int(dist/(np.median(dists)/2) + 0.5) - 1
            else:
                n_p = int(dist/(mean_dist/2) + 0.5) - 1
            if abs(slope - slope_field) < 0.03 and n_p > 0:
                missed_points.append([(p1[0] + p2[0])/ 2, (p1[1] + p2[1])/2])
    return missed_points

"""
Determine the slopes and distances between two Voronoi points which have Voronoi_polygons which are adjacent and flagged as large
(and no other adjacent polygons are flagged as large). This information is used for finding missed points in
adjacent_missed_regions which are larger that 2.

Parameters
-----------
vor : Voronoi_diagram
    The Voronoi diagram containing it all
adjacent_missed_regions : list of set of Voronoi_polygon
    Contains a set of Voronoi_polygons which are flagged as large and all are adjacent

Returns
-----------
slopes : list of float
    slopes between two points in adjacent_missed_regions which have length 2
dists : list of float
    slopes between two points in adjacent_missed_regions which have length 2
"""
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

"""
Determine missed points in sets of large Voronoi regions which length longer than 2.
First Voronoi points in a line are detected with the slope_field parameter. Then for each
line missed points are inserted for which the number depends on the dists list found in 
get_slopes_and_distances_in_pairs_of_large_regions.

Parameters
-----------
adjacent_missed_regions : list of set of Voronoi_polygon
    Contains a set of Voronoi_polygons which are flagged as large and all are adjacent
vor : Voronoi_diagram
    The Voronoi diagram containing it all
slope_field : float
    The flope of the field in radians
dists : list of float
    List of distances between pairs of large regions
spinedex : pyqtree Index
    Quad tree for the Voronoi points, if a missed point found is too close to a point in spindex it is ignored
first_it : Boolean
    Whether this function is trigged in the first iteration, if so mean_dist is used instead of dists
mean_dist : float
    The mean of all dists list from the first iteration

Returns
-----------
missed_points : list of coordinates
    All missed points found with this method
"""
def find_missed_points_in_regions(adjacent_missed_regions, vor, slope_field, dists, spindex, first_it=False, mean_dist=None):
    missed_points = []
    for i in range(len(adjacent_missed_regions)):
        if len(adjacent_missed_regions[i]) != 2:
            l = list(adjacent_missed_regions[i])
            lines = util.find_points_in_line(l, slope_field, vor)
            for line in lines:
                for c in range(len(line) - 1):
                    dist = np.sqrt((line[c][1] - line[c + 1][1])**2 + (line[c][0] - line[c + 1][0])**2)
                    if first_it:
                        n_p = int(dist/(np.nanmedian(dists)/2) + 0.5) - 1
                    else:
                        n_p = int(dist/(mean_dist/2) + 0.5) - 1
                    if n_p > 0:
                        if first_it:
                            mps = util.fill_points_in_line(line[c], line[c + 1], n_p, spindex, np.median(dists)/4)
                        else:
                            mps = util.fill_points_in_line(line[c], line[c + 1], n_p, spindex, mean_dist/4)
                        for mp in mps:
                            if mp not in missed_points:
                                missed_points.append(mp)
    return missed_points

"""
Find large Voronoi_polygons which are adjacent and group them together.

Parameters
-----------
ps : list of Voronoi_polygon
    All Voronoi_polygons which are flagged as large

Returns
-----------
aps : list of set of Voronoi_polygon
    The original large Voronoi_polygons grouped together in sets
"""
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

"""
Plot the Voronoi_diagram

Parameters
-----------
vor : Voronoi_diagram
    The Voronoi_diagram to plot
plot_points : list of coordinates
    List of points to plot
missed_regions : list of Voronoi_polygon
    All Voronoi_polygons which are flagged as large
small_regions : list of Voronoi_polygon
    All Voronoi_polygons which are flagged as small
show_vertices : Boolean
    Whether to show Voronoi vertices, default is False
"""
def plot_voronoi_diagram(vor, plot_points, missed_regions, small_regions, show_vertices=False):
    voronoi_plot_2d(vor, show_vertices=show_vertices)
    if len(plot_points) > 0:
        plt.scatter(plot_points[:,0], plot_points[:,1], color='k')
    for r in missed_regions:
        plt.fill(*zip(*r.vc), color="r", alpha = 0.5)
        
    for r in small_regions:
        plt.fill(*zip(*r.vc), color="g", alpha = 1)
        
    plt.show()