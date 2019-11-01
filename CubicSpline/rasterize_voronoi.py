import gdal
import cv2
import numpy as np
from shapely.geometry import Polygon, GeometryCollection, mapping, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm
import geopandas as gpd
import ogr
import multiprocessing as mp
from collections import OrderedDict, Counter

vector_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_count\voronoi_polygons3.shp"
raster_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201907010933-GR_clustering_output.tif"
target_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201907010933-GR_burned_voronoi.tif"

n_processes = 4

n_pixels = 100 #contours with area 100 pixels are ignored, use 100 for medium to large crops, 50 for small

# =============================================================================
# raster_ds = gdal.Open(raster_layer, gdal.GA_ReadOnly)
# xs = raster_ds.RasterXSize
# ys = raster_ds.RasterYSize
# gt = raster_ds.GetGeoTransform()
# proj = raster_ds.GetProjection()
# 
# driver = gdal.GetDriverByName('GTiff')
# target_ds = driver.Create(target_layer, xs, ys, 1, gdal.GDT_Int32, options=['COMPRESS=LZW'])
# target_ds.SetGeoTransform(gt)
# target_ds.SetProjection(proj)
# source_ds = ogr.Open(vector_layer)
# source_layer = source_ds.GetLayer()
# 
# ds = gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=index"])
# 
# target_ds = 0
# =============================================================================


def getCoordByIndices(x, y, gt):
    return gt[0] + y * gt[1], gt[3] + x * gt[5]
    
def get_intersections(offs, array_p, array_m, vfunc, gt):
    polys = []
    indices = []
    if len(array_p) > 0 and len(array_m) > 0:
        array = array_p * array_m
        unique_values = np.unique(array)
        temp = np.zeros(array.shape, dtype=np.uint8)
        for u in unique_values:
            if u != 0:
                temp[array == u] = 1
                _, contours, hier = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
                p = dict()
                for i,c in enumerate(contours):
                    if len(c) > 2 and Polygon(p[0] for p in c).area > n_pixels:
                        coords = vfunc(*zip(np.array([[p[0][0] + offs[0], p[0][1] + offs[1]][::-1] for p in c]).T), gt=gt)
                        coords = np.array([coords[0][0], coords[1][0]])
                        if hier[0,i,3] == -1:
                            p[i] = dict()
                            p[i]['exterior'] = coords
                        else:
                            if 'interior' in p[hier[0,i,3]]:
                                p[hier[0,i,3]]['interior'].append(coords)
                            else:
                                p[hier[0,i,3]]['interior'] = [coords]
                if p:
                    pp = []
                    for key in p.keys():
                        pp.append(Polygon(np.array(p[key]['exterior']).T, [np.array(p[key]['interior'][i]).T for i in range(len(p[key]['interior']))] if 'interior' in p[key] else []))
                    if len(pp) > 1:
                        poly = MultiPolygon(pp)
                    else:
                        poly = pp[0]
                
                    if not poly.is_valid:
                        poly_b = poly.buffer(0)
                        if not poly_b.is_empty and poly_b.area > poly.area/2:
                            polys.append(poly_b)
                        else:
                            polys.append(poly)
                    else:
                        polys.append(poly)

                    indices.append(u)
                temp = np.zeros(array.shape, dtype=np.uint8)
    return polys, indices

#%%
if __name__ == "__main__":
    blocksize = 2000
    ds = gdal.Open(target_layer, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()   
    ds_mask = gdal.Open(raster_layer, gdal.GA_ReadOnly)
    vfunc = np.vectorize(getCoordByIndices, excluded=['gt'])
    
    xsize = ds.GetRasterBand(1).XSize
    ysize = ds.GetRasterBand(1).YSize
    
    cols = int(np.ceil(xsize / blocksize))
    rows = int(np.ceil(ysize / blocksize))
    offsets = []
    
    for i in range(cols):
        for j in range(rows):
            xoff = blocksize + 2 if blocksize * (i + 1) < xsize else xsize - blocksize * i
            yoff = blocksize + 2 if blocksize * (j + 1) < ysize else ysize - blocksize * j
            offsets.append((blocksize * i  - 2 if i != 0 else blocksize * i, blocksize * j - 2 if j !=0 else blocksize * j, xoff, yoff))
            
    arrays_p = []
    arrays_m = []
    offsets_with_data = []
    pbar0 = tqdm(total=len(offsets), desc='loading arrays', position=0)
    for offset in offsets:
        try:
            arrays_p.append(ds.GetRasterBand(1).ReadAsArray(*offset))
            try:
                arrays_m.append(ds_mask.GetRasterBand(1).ReadAsArray(*offset))
                offsets_with_data.append(offset)
            except:
                pass
        except:
            pass
        
        pbar0.update(1)
    pbar0.close()
    pbar = tqdm(total=len(offsets_with_data), desc='finding contours', position=0)
    p = mp.Pool(n_processes)
    results = [p.apply_async(get_intersections, (offsets_with_data[i], arrays_p[i], arrays_m[i], vfunc, gt,)) for i in range(len(offsets_with_data))]
    
    polys = []
    indices = []
    for i in range(len(offsets_with_data)):
        polys.extend(results[i].get()[0])
        indices.extend(results[i].get()[1])
        pbar.update(1)
    pbar.close()
    
    n_occs = Counter(indices)
    unique_values = [(i,j) for i,j in enumerate(indices) if n_occs[j] > 1]
    unified_polys = dict()
    pbar2 = tqdm(total=len(unique_values), desc="find polys to dissolve", position=0)
    for i in range(len(unique_values)-1, -1, -1):
        if unique_values[i][1] not in unified_polys.keys():
            unified_polys[unique_values[i][1]] = [polys[unique_values[i][0]]]
        else:
            unified_polys[unique_values[i][1]].append(polys[unique_values[i][0]])
        pbar2.update(1)
    pbar2.close()
    
    
    for i in range(len(unique_values) -1, -1, -1):
        del polys[unique_values[i][0]]
        del indices[unique_values[i][0]]
        
    pbar3 = tqdm(total=len(unified_polys), desc="dissolving polys", position=0)
    for key in unified_polys.keys():
        try:
            polys.append(unary_union(unified_polys[key]))
            indices.append(key)
        except:
            polys.append(MultiPolygon(unified_polys[key]))
            indices.append(key)
        pbar3.update(1)
    pbar3.close()
    
    df = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(polys), 'nr': indices})
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs({'init': 'epsg:28992'})
    df.to_file("_".join(raster_layer.split("_")[:-1]) + "_voronoi.shp")

