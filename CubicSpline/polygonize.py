import geopandas
from shapely.geometry import Polygon, MultiPolygon
from shapely.prepared import prep
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
import util_voronoi as utilv

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
    pbar0 = tqdm(total = 1, desc='finding contours', position=0)
    _, contours, hier = cv.findContours(array.astype(np.uint8), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
    pbar0.update(1)
    pbar0.close()

    vfunc = np.vectorize(getCoordByIndices)
    
    pbar = tqdm(total=(len(contours)), desc="creating polys from contours", position=0)
    polys = OrderedDict()
    for i, c in enumerate(contours):
        if len(c) > 2 and Polygon([p[0] for p in c]).area > n_pixels:
            poly = Polygon(np.array(vfunc([p[0][1] for p in c], [p[0][0] for p in c])).T)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if isinstance(poly, MultiPolygon):
                poly = sorted(list(poly), key=lambda a:a.area, reverse=True)[0]
            if hier[0,i,3] == -1:
                if str(i) in polys:
                    polys[str(i)]['boundary'] = poly
                else:
                    polys[str(i)] = {'boundary': None, 'holes': [], 'vor_polys': []}
                    polys[str(i)]['boundary'] = poly
            else:
                if str(hier[0,i,3]) in polys:
                    polys[str(hier[0,i,3])]['holes'].extend([poly])
                else:
                    polys[str(hier[0,i,3])] = {'boundary': None, 'holes': [], 'vor_polys': []}
                    polys[str(hier[0,i,3])]['holes'] = [poly]
        pbar.update(1)
    pbar.close()
    
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
            
    ordered = dict(OrderedDict(sorted(polys.items(), key = lambda p:p[1]['boundary'].area, reverse=True)))
    pbar2 = tqdm(total=len(vor_polys), desc="sorting")
    ordered_intersecting_vor_polys = []
    ordered_contained_vor_polys = []
    index_i = []
    index_c = []
    for key in ordered.keys():
        prepped = prep(ordered[key]['boundary'])
        contained_vor_polys = list(filter(lambda p: prepped.contains(p), vor_polys))
        ordered_contained_vor_polys += contained_vor_polys
        ordered[key]['vor_polys_c'] = contained_vor_polys
        index_c += [key for _ in range(len(contained_vor_polys))]
        vor_polys = list(filter(lambda p : not prepped.contains(p), vor_polys))
        intersecting_vor_polys = list(filter(lambda p : prepped.intersects(p), vor_polys))
        ordered_intersecting_vor_polys += intersecting_vor_polys
        index_i += [key for _ in range(len(intersecting_vor_polys))]
        vor_polys = list(filter(lambda p: not prepped.intersects(p), vor_polys))
        ordered[key]['vor_polys_i'] = intersecting_vor_polys
        pbar2.update(len(contained_vor_polys)+len(intersecting_vor_polys))
    pbar2.close()
        
    p = mp.Pool(n_processes)
    batches = []
    batchsize = int(len(ordered_intersecting_vor_polys) / n_batches)
    results = [p.apply_async(intersect, (polys, index_i[i*batchsize: (i+1)*batchsize], ordered_intersecting_vor_polys[i*batchsize: (i+1)*batchsize])) for i in range(n_batches)]
    
    ints = []
    pbar3 = tqdm(total=len(results), desc="intersecting vor polys on boundary", position=0)   
    for i in range(len(results)):
        ints += results[i].get()
        pbar3.update(1)
    pbar3.close()
    
    pbar4 = tqdm(total=len(polys.keys()), desc="intersecting with holes", position = 0)
    for key in polys.keys():
        if polys[key]['holes']:
            t = STRtree(polys[key]['holes'])
            for i in range(len(ordered_contained_vor_polys)):
                if index_c[i] == key:
                    for hole in t.query(ordered_contained_vor_polys[i]):
                        if hole.overlaps(ordered_contained_vor_polys[i]):
                            try:
                                ordered_contained_vor_polys[i] = ordered_contained_vor_polys[i].difference(hole)
                            except:
                                pass
            for i in range(len(ints)):
                if index_c[i] == key:
                    for hole in t.query(ints[i]):
                        if hole.overlaps(ints[i]):
                            try:
                                ints[i] = ints[i].difference(hole)
                            except:
                                pass
        pbar4.update(1)
    pbar4.close()
            
    dfv = geopandas.GeoDataFrame({'geometry': ints + ordered_contained_vor_polys})
    dfv.crs = {'init': 'epsg:4326'}
    dfv = dfv.to_crs({'init': 'epsg:28992'})
    dfv.to_file("_".join(mask_path.split("_")[:-1])+"_voronoi.shp")