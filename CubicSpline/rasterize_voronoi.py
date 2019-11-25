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
import datetime
import voronoi2shp

def getCoordByIndices(x, y, gt):
    return gt[0] + y * gt[1], gt[3] + x * gt[5]

def get_intersections(offs, array_p, array_m, vfunc, gt, n_pixels):
    polys = []
    indices = []
    if len(array_p) > 0 and len(array_m) > 0:
        array = np.multiply(array_p,array_m)
        unique_values = np.unique(array)
        temp = np.zeros(array.shape, dtype=np.uint8)
        for u in unique_values:
            if u != 0:
                temp[array == u] = 1
                contours, hier = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
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

  
#vector_layer = 
#raster_layer = r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\LAI\c01_verdonk-Wever west-201907170749-GR_clustering_output.tif"
#target_layer = r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\c01_verdonk-Wever west-201907170749-GR_clustering_voronoi_burned.tif"

def create_index_column(vector_layer):
    gdf = gpd.read_file(vector_layer)
    if 'index' not in gdf.columns:
        gdf['index'] = gdf.index + 1    
        gdf.to_file(vector_layer)
        return print('index column created')
    else:
        return

def create_plant_polygons(vector_layer, raster_layer, target_layer):
    #vector_layer = voronoi2shp.create_voronoi_from_points(vector_layer)
    create_index_column(vector_layer)
    
    n_processes = 4
    
    n_pixels = 50 #contours with area 100 pixels are ignored, use 100 for medium to large crops, 50 for small
    
    # =============================================================================
    raster_ds = gdal.Open(raster_layer, gdal.GA_ReadOnly)
    xs = raster_ds.RasterXSize
    ys = raster_ds.RasterYSize
    gt = raster_ds.GetGeoTransform()
    proj = raster_ds.GetProjection()
    
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(target_layer, xs, ys, 1, gdal.GDT_Int32, options=['COMPRESS=LZW'])
    target_ds.SetGeoTransform(gt)
    target_ds.SetProjection(proj)
    source_ds = ogr.Open(vector_layer)
    source_layer = source_ds.GetLayer()
    
    ds = gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=index"])
    
    raster_ds = None
    target_ds = None
    source_ds = None
    ds = None
# =============================================================================



#%%
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
    for offset in offsets[:]:
        try:
            arrays_p.append(ds.GetRasterBand(1).ReadAsArray(*offset))
            try:
                arr = ds_mask.GetRasterBand(1).ReadAsArray(*offset).astype(np.uint8)
                if ds_mask.GetRasterBand(1).GetNoDataValue() == 250:
                    arr[arr == 250] = 0
                    arr[arr > 0] = 1
                if 25 in arr:
                    arr[arr < 26] = 0
                    arr[arr > 26] = 1
                elif 125 in arr:
                    arr[arr == 125] = 0
                    arr[arr > 125] = 1
                arrays_m.append(arr)
                #arrays_m.append(ds_mask.GetRasterBand(1).ReadAsArray(*offset))
                offsets_with_data.append(offset)
            except:
                pass
        except:
            pass

    ds = None
    ds_mask = None

    pbar0.update(1)
    pbar0.close()
    pbar = tqdm(total=len(offsets_with_data), desc='finding contours', position=0)
    p = mp.Pool(n_processes)
    results = [p.apply_async(get_intersections, (offsets_with_data[i], arrays_p[i], arrays_m[i], vfunc, gt, n_pixels)) for i in range(len(offsets_with_data))]

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
    
    try:
        df = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(polys), 'nr': indices})
        df.crs = {'init': 'epsg:4326'}
        #df = df.to_crs({'init': 'epsg:28992'})
        df.to_file("_".join(raster_layer.split("_")[:-1]) + "_voronoi.shp")
    except:
        pass
    return print("finished run of " + target_layer + " at " + str(datetime.datetime.now()))

if __name__=='__main__':
    vector_layers = [r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Wever west\20190717\0749\Plant_count\20190717_count_voronoi_polys.shp"
                    #r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Rijweg stalling 1\20190717\0849\Plant_count\20190717_count.shp",
                    #r"D:\800 Operational\c08_biobrass\AZ74\plant_count.shp",
                    #r"D:\800 Operational\c08_biobrass\AZ91\plant_count.shp"
                    ]
    
    for vector_layer in vector_layers:
        if vector_layer == r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Wever west\20190717\0749\Plant_count\20190717_count_voronoi_polys.shp":
            raster_layers = [r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\LAI\c01_verdonk-Wever west-201908291238-GR_clustering_output_.tif"]#,
                            #r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\LAI\c01_verdonk-Wever west-201907170749-GR_clustering_output.tif",
                            #r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\LAI\c01_verdonk-Wever west-201907240724-GR_clustering_output.tif",
                            #r"D:\800 Operational\c01_verdonk\Wever west\Season evaluation\LAI\c01_verdonk-Wever west-201908041528-GR_clustering_output.tif"]
            for raster_layer in raster_layers:
                target_layer = raster_layer[:-4] + "_voronoi.tif"
                create_plant_polygons(vector_layer, raster_layer, target_layer)
                
            
        elif vector_layer == r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Rijweg stalling 1\20190717\0849\Plant_count\20190717_count.shp":
            raster_layers = [r"D:\800 Operational\c01_verdonk\Rijweg stalling 1\Season evaluation\LAI\c01_verdonk-Rijweg stalling 1-201907170849-GR_clustering_output.tif",
                            r"D:\800 Operational\c01_verdonk\Rijweg stalling 1\Season evaluation\LAI\c01_verdonk-Rijweg stalling 1-201907230859-GR_clustering_output.tif",
                            r"D:\800 Operational\c01_verdonk\Rijweg stalling 1\Season evaluation\LAI\c01_verdonk-Rijweg stalling 1-201908041120-GR_clustering_output.tif",
                            r"D:\800 Operational\c01_verdonk\Rijweg stalling 1\Season evaluation\LAI\c01_verdonk-Rijweg stalling 1-201907091137-GR_clustering_output.tif"]
            for raster_layer in raster_layers:
                target_layer = raster_layer[:-4] + "_voronoi.tif"
                create_plant_polygons(vector_layer, raster_layer, target_layer)
            
        elif vector_layer == r"D:\800 Operational\c08_biobrass\AZ74\plant_count.shp":
            raster_layers = [r"D:\800 Operational\c08_biobrass\AZ74\Season evaluation\LAI\c08_biobrass-AZ74-201905291141-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ74\Season evaluation\LAI\c08_biobrass-AZ74-201906041446-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ74\Season evaluation\LAI\c08_biobrass-AZ74-201906141536-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ74\Season evaluation\LAI\c08_biobrass-AZ74-201905171650-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ74\Season evaluation\LAI\c08_biobrass-AZ74-201905271248-GR_clustering_output.tif"]
            for raster_layer in raster_layers:
                target_layer = raster_layer[:-4] + "_voronoi.tif"
                create_plant_polygons(vector_layer, raster_layer, target_layer)
                
        elif vector_layer == r"D:\800 Operational\c08_biobrass\AZ91\plant_count.shp":
            raster_layers = [r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201906281013-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201907050837-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201907161410-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201905271321_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201906141505-GR_clustering_output.tif",
                            r"D:\800 Operational\c08_biobrass\AZ91\Season evaluation\LAI\c08_biobrass-AZ91-201906201457-GR_clustering_output.tif"]
            for raster_layer in raster_layers:
                target_layer = raster_layer[:-4] + "_voronoi.tif"
                create_plant_polygons(vector_layer, raster_layer, target_layer)
    
