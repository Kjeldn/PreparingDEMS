import geopandas
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep
from shapely.strtree import STRtree
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import Voronoi
from tqdm import tqdm
import warnings
import cv2 as cv
import gdal
from tkinter import filedialog
from tkinter import *
from collections import OrderedDict
warnings.filterwarnings("ignore")
import multiprocessing as mp
import util_voronoi as utilv
import polygonize
from pyqtree import Index

n_processes = 4 #number of processes used for intersecting vor polys
n_batches = 40 #number of batches used for intersecting vor polys
n_pixels = 50 #number of pixels of an interior of an contour to be ignored

def getCoordByIndices(x, y):
    return gt[0] + y * gt[1], gt[3] + x * gt[5]

def intersect(polys, index_i, ordered_intersecting_vor_polys):
    for i in range(len(ordered_intersecting_vor_polys)):
        try:
            ordered_intersecting_vor_polys[i] = ordered_intersecting_vor_polys[i].intersection(polys[index_i[i]]['boundary'])
        except:
            pass
    return ordered_intersecting_vor_polys

if __name__ == "__main__":
    root = Tk()
    mask_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select mask raster", parent=root)
    plant_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select plants shape file", parent=root)
    root.destroy()
    file = gdal.Open(mask_path)
    band = file.GetRasterBand(1)
    gt = file.GetGeoTransform()
    array = band.ReadAsArray()
    polys = polygonize.get_contour_polygons_seperated(n_pixels, array, gt)
    
    holes = []
    holes_i = []
    for key in polys.keys():
        if polys[key]['holes']:
            holes += polys[key]['holes']
            holes_i += [key for _ in range(len(polys[key]['holes']))]
    
    plants = geopandas.read_file(plant_path)
    plants_r, mx, my = utilv.readable_values(np.array([(p.x, p.y) for p in plants.loc[:,'geometry']]))
    vor = Voronoi(plants_r)
    convex_hull = Polygon([(p.x, p.y) for p in plants.loc[:,'geometry']]).convex_hull
    vor_polys = []
    for r in vor.regions:
        if -1 not in r and len(r) > 2 and Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)).within(convex_hull):
            vor_polys.append(Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)))
    
    polys_list = []
    for key in polys.keys():
        if isinstance(polys[key]['boundary'], Polygon) and polys[key]['boundary'].exterior:
            polys_list.append(polys[key]['boundary'])
        elif isinstance(polys[key]['boundary'], MultiPolygon):
            for poly in polys[key]['boundary']:
                polys_list.append(poly)
                
    ordered = dict(OrderedDict(sorted(polys.items(), key = lambda p:p[1]['boundary'].area, reverse=True)))
                
    points_list = [[] for _ in range(len(polys))]
    
    spindex = Index(Polygon(utilv.readable_values_inv(vor.points, mx, my)).bounds)
    for i, p in enumerate(utilv.readable_values_inv(vor.points, mx, my)):
        spindex.insert({'index': i, 'coord': p}, (p[0], p[1], p[0], p[1]))

    pbar15 = tqdm(total=len(ordered), desc="sorting", position=0)
    ints = []
    for i, key in enumerate(ordered.keys()):
        if (isinstance(ordered[key]['boundary'], Polygon) and ordered[key]['boundary'].exterior) or isinstance(ordered[key]['boundary'], MultiPolygon):
            prepped = prep(ordered[key]['boundary'])
            filtered = list(filter(lambda a : prepped.contains(Point(a['coord'][0], a['coord'][1])), spindex.intersect(ordered[key]['boundary'].bounds)))
            points_list[i].extend([p['index'] for p in filtered])   
        pbar15.update(1)
    pbar15.close()
     
    pbar2 = tqdm(total=len(ordered.keys()), desc='sorting more', position=0)
    for i, key in enumerate(ordered.keys()):
        if (isinstance(ordered[key]['boundary'], Polygon) and ordered[key]['boundary'].exterior) or isinstance(ordered[key]['boundary'], MultiPolygon):
            prepped = prep(ordered[key]['boundary'])
            prepped_exterior = prep(ordered[key]['boundary'].exterior if isinstance(ordered[key]['boundary'], Polygon) else unary_union([poly.exterior for poly in ordered[key]['boundary']]))
            ordered[key]['vor_polys_c'] = list(filter(prepped.contains, [Polygon(utilv.readable_values_inv(vor.vertices[vor.regions[vor.point_region[p]]], mx, my)) for p in points_list[i] if -1 not in vor.regions[vor.point_region[p]]]))
            ordered[key]['vor_polys_i'] = list(filter(prepped_exterior.intersects, [Polygon(utilv.readable_values_inv(vor.vertices[vor.regions[vor.point_region[p]]], mx, my)) for p in points_list[i] if -1 not in vor.regions[vor.point_region[p]]]))  
        pbar2.update()
    pbar2.close()

    
e = next(iter(ordered))
for poly in ordered[e]['vor_polys_c']:
    plt.plot(*poly.exterior.xy, 'b')
for poly in ordered[e]['vor_polys_i']:
    plt.plot(*poly.exterior.xy, 'r')
    
for poly in ordered[e]['boundary']:
    plt.plot(*poly.exterior.xy, 'g')