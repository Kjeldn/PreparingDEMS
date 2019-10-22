import gdal
import cv2
import numpy as np
from shapely.geometry import Polygon, GeometryCollection
from tqdm import tqdm
import geopandas as gpd
import multiprocessing as mp

vector_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_count\voronoi_polygons3.shp"
raster_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906250739-GR_clustering_output.tif"
target_layer = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906250739-GR_burned_voronoi.tif"

n_processes = 4

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
        unique_values = np.delete(unique_values, 0, 0)  
        
        for u in unique_values:
            temp = np.array(array, copy=True)
            temp[temp != u] = 0
            temp[temp == u] = 1
            _, contours, hier = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c= sorted(contours, key=len, reverse=True)[0]
            if len(c) > 2:
                coords = vfunc(*zip(np.array([[p[0][0] + offs[0], p[0][1] + offs[1]][::-1] for p in c]).T), gt=gt)
                poly = Polygon(np.array([coords[0][0], coords[1][0]]).T)
    
                if poly.is_valid:
                    polys.append(poly)
                else:
                    poly = poly.buffer(0)
                    polys.append(poly)
                indices.append(u)

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
            xoff = blocksize if blocksize * (i + 1) < xsize else xsize - blocksize * i
            yoff = blocksize if blocksize * (j + 1) < ysize else ysize - blocksize * j
            offsets.append((blocksize * i, blocksize * j, xoff, yoff))
            
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
    for i in range(len(offsets)):
        polys.extend(results[i].get()[0])
        indices.extend(results[i].get()[1])
        pbar.update(1)
    pbar.close()
    
    df = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(polys), 'index': indices})
    
    dissolved = df.dissolve(by='index', aggfunc='first')
    for i in dissolved.index:
        if dissolved.loc[i,'geometry'].type == 'GeometryCollection':
            dissolved = dissolved.drop(i)
        
    dissolved.to_file("_".join(raster_layer.split("_")[:-1]) + "_voronoi.shp")
