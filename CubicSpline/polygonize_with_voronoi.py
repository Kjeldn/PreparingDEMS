import geopandas
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep
from shapely.ops import unary_union
from shapely.strtree import STRtree
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
import polygonize
from pyqtree import Index
import util_voronoi as utilv
from itertools import islice
from random import shuffle

n_processes = 4 #number of processes used for intersecting vor polys
n_batches = 1000 #number of batches used for intersecting vor polys
n_pixels = 50 #number of pixels of an interior of an contour to be ignored

def divide_into_batches(d, n):
    dicts = [dict() for _ in range(n)]
    for i, key in enumerate(d.keys()):
        dicts[i % n][key] = d[key]
    shuffle(dicts)
    return dicts

def intersect(polys_dict):
    for key in polys_dict.keys():
        if 'vor_polys_i' in polys_dict[key].keys():
            for i in range(len(polys_dict[key]['vor_polys_i'])):
                try:
                    polys_dict[key]['vor_polys_i'][i] = polys_dict[key]['vor_polys_i'][i].intersection(polys_dict[key]['boundary'])
                except:
                    try:
                        polys_dict[key]['vor_polys_i'][i] = polys_dict[key]['vor_polys_i'][i].buffer(0).intersection(polys_dict[key]['boundary'])
                    except:
                        pass
    return polys_dict

def readable_values_inv(x, y, mean_x_coord, mean_y_coord):
    f = 10000
    return x / f + mean_x_coord, y / f + mean_y_coord

if __name__ == "__main__":
    #root = Tk()
    #mask_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select mask raster", parent=root)
    #plant_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select plants shape file", parent=root)
    #root.destroy()
    mask_path = r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\archive\c01_verdonk-Wever west-201907170749-GR_clustering_output.tif"
    plant_path= r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Wever west\20190717\0749\Plant_count\20190717_count.shp"
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
    plants = plants.mask(plants.geometry.eq('None')).dropna()
    plants_r, mx, my = utilv.readable_values(np.array([(p.centroid.xy[0][0], p.centroid.xy[1][0]) for p in plants.loc[:,'geometry']]))
    vor = Voronoi(plants_r)
    convex_hull = Polygon([(p.centroid.xy[0][0], p.centroid.xy[1][0]) for p in plants.loc[:,'geometry']]).convex_hull
    vor_polys = []
    for r in vor.regions:
        if -1 not in r and len(r) > 2 and Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)).within(convex_hull):
            vor_polys.append(Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)))
            
    ordered = dict(OrderedDict(sorted(polys.items(), key = lambda p:p[1]['boundary'].area, reverse=True)))
# =============================================================================
#     pbar2 = tqdm(total=len(vor_polys), desc="sorting", position = 0)
#     ordered_intersecting_vor_polys = []
#     ordered_contained_vor_polys = []
#     index_i = []
#     index_c = []
#     for key in ordered.keys():
#         prepped = prep(ordered[key]['boundary'])
#         contained_vor_polys = list(filter(lambda p: prepped.contains(p), vor_polys))
#         ordered_contained_vor_polys += contained_vor_polys
#         ordered[key]['vor_polys_c'] = contained_vor_polys
#         index_c += [key for _ in range(len(contained_vor_polys))]
#         vor_polys = list(filter(lambda p : not prepped.contains(p), vor_polys))
#         intersecting_vor_polys = list(filter(lambda p : prepped.intersects(p), vor_polys))
#         ordered_intersecting_vor_polys += intersecting_vor_polys
#         index_i += [key for _ in range(len(intersecting_vor_polys))]
#         vor_polys = list(filter(lambda p: not prepped.intersects(p), vor_polys))
#         ordered[key]['vor_polys_i'] = intersecting_vor_polys
#         pbar2.update(len(contained_vor_polys)+len(intersecting_vor_polys))
#     pbar2.close()
# =============================================================================
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
    
    vread_values = np.vectorize(readable_values_inv, excluded=["mean_x_coord", "mean_y_coord"])
    pbar2 = tqdm(total=len(ordered.keys()), desc='sorting more', position=0)
    for i, key in enumerate(ordered.keys()):
        if (isinstance(ordered[key]['boundary'], Polygon) and ordered[key]['boundary'].exterior) or isinstance(ordered[key]['boundary'], MultiPolygon):
            prepped = prep(ordered[key]['boundary'])
            prepped_exterior = prep(ordered[key]['boundary'].exterior
                                    if isinstance(ordered[key]['boundary'], Polygon)
                                    else unary_union([poly.exterior for poly in ordered[key]['boundary']]))
            ordered[key]['vor_polys_c'] = list(filter(
                    prepped.contains, 
                    [Polygon(np.array(vread_values(
                            x=np.array(vor.vertices[vor.regions[vor.point_region[p]]])[:,0], 
                            y=np.array(vor.vertices[vor.regions[vor.point_region[p]]])[:,1], 
                            mean_x_coord=mx, mean_y_coord=my)).T)
                    for p in points_list[i] if -1 not in vor.regions[vor.point_region[p]]]))
            ordered[key]['vor_polys_i'] = list(filter(
                    prepped_exterior.intersects, 
                    [Polygon(np.array(vread_values(
                            x=np.array(vor.vertices[vor.regions[vor.point_region[p]]])[:,0], 
                            y=np.array(vor.vertices[vor.regions[vor.point_region[p]]])[:,1], 
                            mean_x_coord=mx, mean_y_coord=my)).T) 
                    for p in points_list[i] if -1 not in vor.regions[vor.point_region[p]]]))
        pbar2.update()
    pbar2.close()
        
    p = mp.Pool(n_processes)
    batches = divide_into_batches(ordered, n_batches)
    results = [p.apply_async(intersect, (batches[i],)) for i in range(n_batches)]
    
    d = dict()
    pbar3 = tqdm(total=len(results), desc="intersecting vor polys on boundary", position=0)   
    for i in range(len(results)):
        for key in results[i].get().keys():
            d[key] = results[i].get()[key]
        pbar3.update(1)
    pbar3.close()
    
    pbar4 = tqdm(total=len(polys.keys()), desc="intersecting with holes", position = 0)
    for key in polys.keys():
        if d[key]['holes']:
            t = STRtree(d[key]['holes'])
            if 'vor_polys_c' in d[key].keys():
                for i in range(len(d[key]['vor_polys_c'])):
                    for hole in t.query(d[key]['vor_polys_c'][i]):
                        if hole.overlaps(d[key]['vor_polys_c'][i]):
                            try:
                                d[key]['vor_polys_c'][i] = d[key]['vor_polys_c'][i].difference(hole)
                            except:
                                pass
            if 'vor_polys_i' in d[key].keys():
                for i in range(len(d[key]['vor_polys_i'])):
                    for hole in t.query(d[key]['vor_polys_i'][i]):
                        if hole.overlaps(d[key]['vor_polys_i'][i]):
                            try:
                                ints[i] = d[key]['vor_polys_i'][i].difference(hole)
                            except:
                                pass
        pbar4.update(1)
    pbar4.close()
    
    ints = []
    for key in d.keys():
        ints.extend(d[key]['vor_polys_i'] + d[key]['vor_polys_c'])
            
    dfv = geopandas.GeoDataFrame({'geometry': ints})
    dfv.crs = {'init': 'epsg:4326'}
    dfv = dfv.to_crs({'init': 'epsg:28992'})
    dfv.to_file("_".join(mask_path.split("_")[:-1])+"_voronoi.shp")