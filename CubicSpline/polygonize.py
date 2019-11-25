"""
Reads binary image, find contours and writes them to a shape file.
CRS of resulting shape file is espg:28992 and location is "_".join(mask_path.split("_")[:-1])+"_polys.shp".
For binary images with small crops n_pixels should be small (~20). When crops are larger and grouped together
n_pixels should be larger (~50), this will ignore the holes in the binary image caused by overexposure.
"""

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
import util_voronoi as utilv

n_pixels = 20 #number of pixels of an interior of an contour to be ignored

def getCoordByIndices(x, y, gt):
    return gt[0] + y * gt[1], gt[3] + x * gt[5]

def intersect(polys, index_i, ordered_intersecting_vor_polys):
    for i in range(len(ordered_intersecting_vor_polys)):
        try:
            ordered_intersecting_vor_polys[i] = ordered_intersecting_vor_polys[i].intersection(polys[index_i[i]]['boundary'])
        except:
            pass
    return ordered_intersecting_vor_polys

def get_contour_polygons_seperated(n_pixels, array, gt):
    pbar0 = tqdm(total = 1, desc='finding contours', position=0)
    contours, hier = cv.findContours(array.astype(np.uint8), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
    pbar0.update(1)
    pbar0.close()

    vfunc = np.vectorize(getCoordByIndices, excluded=['gt'])
    
    pbar = tqdm(total=(len(contours)), desc="creating polys from contours", position=0)
    polys = OrderedDict()
    for i, c in enumerate(contours):
        if len(c) > 2 and Polygon([p[0] for p in c]).area > n_pixels:
            poly = Polygon(np.array(vfunc(x=[p[0][1] for p in c], y=[p[0][0] for p in c], gt=gt)).T)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if hier[0,i,3] == -1:
                if str(i) in polys:
                    polys[str(i)]['boundary'] = poly
                else:
                    polys[str(i)] = {'boundary': None, 'holes': [], 'vor_polys': []}
                    polys[str(i)]['boundary'] = poly
            else:
                if str(hier[0,i,3]) in polys:
                    polys[str(hier[0, i, 3])]['holes'].extend([poly])
                else:
                    polys[str(hier[0, i, 3])] = {'boundary': None, 'holes': [], 'vor_polys': []}
                    polys[str(hier[0, i, 3])]['holes'] = [poly]
        pbar.update(1)
    pbar.close()
    return polys

def get_contour_polygons(n_pixels, array, gt):
    polys = get_contour_polygons_seperated(n_pixels, array, gt)
    polys_list = []
    for key in polys.keys():
        if isinstance(polys[key]['boundary'], Polygon) and polys[key]['boundary'].exterior:
            if polys[key]['holes']:
                polys_list.append(Polygon(list(polys[key]['boundary'].exterior.coords), [list(hole.exterior.coords) for hole in polys[key]['holes']]))
            else:
                polys_list.append(polys[key]['boundary'])
        elif isinstance(polys[key]['boundary'], MultiPolygon):
            for poly in polys[key]['boundary']:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                holes = list(filter(poly.contains, polys[key]['holes']))
                if isinstance(poly, Polygon):
                    if holes:
                        polys_list.append(Polygon(list(poly.exterior.coords), [list(hole.exterior.coords) for hole in holes if isinstance(hole, Polygon)]))
                    else:
                        polys_list.append(poly)
                elif isinstance(poly, MultiPolygon):
                    for polyy in poly:
                        if holes:
                            polys_list.append(Polygon(list(polyy.exterior.coords), [list(hole.exterior.coords) for hole in holes]))
                        else:
                            polys_list.append(polyy)
    return polys_list

if __name__ == "__main__":
    root = Tk()
    mask_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select mask raster", parent=root)
    root.destroy()
    file = gdal.Open(mask_path)
    band = file.GetRasterBand(1)
    gt = file.GetGeoTransform()
    array = band.ReadAsArray()
    polys_list = get_contour_polygons(n_pixels, array, gt)
            
    df = geopandas.GeoDataFrame({'geometry': geopandas.GeoSeries(polys_list)})
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs({'init': 'epsg:28992'})
    df.to_file("_".join(mask_path.split("_")[:-1])+"_polys.shp")