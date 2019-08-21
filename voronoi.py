import gdal
import fiona
import util
import numpy as np
from scipy import interpolate
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import scipy.stats as st

wd = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path = "c01_verdonk-Wever oost-201908041528_DEM-GR"
path_ahn = None #"m_19fn2.tif"


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
        return 12345 * i * len(self.vi)
    
    def is_adjacent(self, other):
        n = 0
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
        
#%% plants
with fiona.open(wd + "/20190717_count.shp") as src:
    n = 2000
    plants = np.zeros((n,2))
    for i in range(n):
        if src[i]['geometry']:
            plants[i, :] = [src[i]['geometry']['coordinates'][0][0], src[i]['geometry']['coordinates'][0][1]]
        
for i in range(plants.shape[1]-1, -1, -1):
    if plants[0][i]==0 and plants[1][i] == 0:
        plants = np.delete(plants, i, 1)
        
plants[:,0] = 10000*(plants[:,0] - np.mean(plants[:,0]))
plants[:,1] = 10000*(plants[:,1] - np.mean(plants[:,1]))

poly = Polygon(zip(plants[:,0], plants[:,1]))
poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
polygon = Polygon(poly_line.coords)
        
vor = Voronoi(plants)


a = []
for i in range(len(vor.regions)):
    if -1 not in vor.regions[i] and vor.regions[i] and len(vor.regions[i]) > 2:
        vs = []
        for j in vor.regions[i]:
            if Point(vor.vertices[j][0], vor.vertices[j][1]).within(polygon):
                vs.append((vor.vertices[j][0], vor.vertices[j][1]))
            else:
                vs = []
                break
        
        if vs:
            poly2 = Polygon(vs)
            a.append(poly2.area)
            
missed_regions = []
            
ci = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale = np.std(a))
for i in range(len(vor.regions)):
    if not -1 in vor.regions[i] and vor.regions[i]:
        vs = []
        for j in vor.regions[i]:
            if Point(vor.vertices[j][0], vor.vertices[j][1]).within(polygon):
                vs.append((vor.vertices[j][0], vor.vertices[j][1]))
            else:
                vs = []
                break
        
        if vs:
            poly2 = Polygon(vs)
            if poly2.area < ci[0] or poly2.area > ci[1]:
                missed_regions.append(Voronoi_polygon(i, vor.regions[i], [list(vor.vertices[k]) for k in vor.regions[i]]))

def find_adjacent_polygons(ps):
    aps = []
    for p in ps:
        for q in ps:
            if p != q and p.is_adjacent(q):
                new = True
                for ap in aps:
                    if p in ap or q in ap:
                        ap.add(p)
                        ap.add(q)
                        new = False
                        break
                if new:
                    aps.append({p, q})
    
    return aps


adjacent_missed_regions = find_adjacent_polygons(missed_regions)
missed_points = []

for s in adjacent_missed_regions:
    if len(s) == 2:
        aps = list(s)
        line = aps[0].adjacent_line(aps[1])
        missed_points.append([(line[0][0] + line[1][0])/ 2, (line[0][1] + line[1][1])/2])

missed_points = np.array(missed_points)

voronoi_plot_2d(vor, show_vertices=False)
plt.scatter(missed_points[:,0], missed_points[:,1], color='g')
for r in missed_regions:
    plt.fill(*zip(*r.vc), color="r", alpha = 0.75)
    
plt.show()