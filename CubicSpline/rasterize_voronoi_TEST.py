import gdal
import cv2
import os
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

## -- End Parameters
#Set 'Yes' if a clip.shp file is used, otherwise it can be set to 'No'.
clip_shpfile_used = 'Yes'
clip_shp_link = r"D:\800 Operational\c07_hollandbean\Season evaluation\clip_shapes\Joke_Visser_clip_shp.shp"
points_chp_link = r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Joke Visser-906031020-GR_plant_count.gpkg"

## -- Start parameters --

def clip_plant_count(clip_shp_link, plant_count_link):
    '''Takes the plant_count points and only returns the points located inside
    the clip_shp.shp
    '''
    clip_shp = gpd.read_file(clip_shp_link)
    points_shp = gpd.read_file(points_chp_link)

    points_check = points_shp['geometry'].intersects(clip_shp['geometry'].unary_union)
    points_inside = points_shp[points_check == True]
    return points_inside

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

def create_plant_polygons(vector_layer, raster_layer, target_layer, p, clip_shp_link, clip_shpfile_used, plant_count_link):

    #clip vector layer with clip_shape. The following parameters are for the
    #clip_plant_count function clip_shp_link, plant_count_link
    if (clip_shpfile_used == 'Yes') | (clip_shpfile_used == 'yes'):
        clipped_vectorlayer = clip_plant_count(clip_shp_link, plant_count_link)
        tempfile_link = "_".join(plant_count_link.split("_")[:-1]) + "tempfile.shp"
        clipped_vectorlayer.to_file(tempfile_link)
        vector_layer = voronoi2shp.create_voronoi_from_points(tempfile_link)
        os.remove(tempfile_link)
    else:
        vector_layer = voronoi2shp.create_voronoi_from_points(vector_layer)

    create_index_column(vector_layer)

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

    blocksize = 2048
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
        pbar0.update(1)
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
    pbar0.close()

    ds = None
    ds_mask = None

    pbar = tqdm(total=len(offsets_with_data), desc='finding contours', position=0)
    #p = mp.Pool(n_processes)
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
    n_processes = 12
    p = mp.Pool(n_processes)
    vector_layers = [r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Joke Visser-906031020-GR_plant_count.gpkg"
            #r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\joke_shp_test_voronoi_polys.shp",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hage-905221113-GR_plant_count._voronoi_polys.shp",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hein de Schutter-201906051255_plant_count_voronoi_polys.shp",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hendrik de Heer-905131422-GR_plant_count_voronoi_polys.shp",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Jos Schelling-906031318-GR_plant_count_voronoi_polys.shp",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\NoviFarm1-906031305-GR_plant_count_voronoi_polys.shp"
                    ]

    for vector_layer in vector_layers:
        if vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Joke Visser-906031020-GR_plant_count.gpkg":
        #r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Joke Visser-906031020-GR_plant_count.gpkg":
            raster_layers = [
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908020829-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907241431-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907101007-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907010933-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906250739-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906191208-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906031020-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905271514-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905221245_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201909060802-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908300729-GR_clustering_output.tif",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908231004-GR_clustering_output.tif"

]
            for raster_layer in raster_layers:
                target_layer = raster_layer[:-4] + "_voronoi.tif"
                create_plant_polygons(vector_layer, raster_layer, target_layer, p, clip_shp_link, clip_shpfile_used, points_chp_link)

#
#        elif vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Jos Schelling-906031318-GR_plant_count_voronoi_polys.shp":
#            raster_layers = [#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201905221346_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201909051036-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201908270803-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201908191407-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201907241542-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201907091020-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201907011300-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201906251102-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201906181027-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201906031318-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Jos Schelling-201905290920-GR_clustering_output.tif"
#]
#            for raster_layer in raster_layers:
#                target_layer = raster_layer[:-4] + "_voronoi.tif"
#                create_plant_polygons(vector_layer, raster_layer, target_layer, p)
#
#        elif vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\NoviFarm1-906031305-GR_plant_count_voronoi_polys.shp":
#            raster_layers = [r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201906251017-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201906180944-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201906031305-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201905271542_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201909051134-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201908271050-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201908191017-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201908021136-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201907241459-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201907090932-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-NoviFarm 1 8ha-201907011229-GR_clustering_output.tif"
#]
#            for raster_layer in raster_layers:
#                target_layer = raster_layer[:-4] + "_voronoi.tif"
#                create_plant_polygons(vector_layer, raster_layer, target_layer, p)
#
#        elif vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hage-905221113-GR_plant_count._voronoi_polys.shp":
#            raster_layers = [#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201905131326-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-20190503_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201909051318-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201908301016-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201908191156-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201907241108-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201907101111-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201907011102-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201906250840-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201906191259-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201906031128-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201905271347-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hage-201905221113-GR_clustering_output.tif"
#]
#            for raster_layer in raster_layers:
#                target_layer = raster_layer[:-4] + "_voronoi.tif"
#                create_plant_polygons(vector_layer, raster_layer, target_layer, p)
#
#        elif vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hein de Schutter-201906051255_plant_count_voronoi_polys.shp":
#            raster_layers = [r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201906241432-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201906171419-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201906051255_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201909061020-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201908291210-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201908231146-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201908061119-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201907240955-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201907081101-GR_clustering_output.tif",
#                            r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hein de Schutter-201907031301-GR_clustering_output.tif"
#                            ]
#            for raster_layer in raster_layers:
#                target_layer = raster_layer[:-4] + "_voronoi.tif"
#                create_plant_polygons(vector_layer, raster_layer, target_layer, p)
#
#        elif vector_layer == r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\Hendrik de Heer-905131422-GR_plant_count_voronoi_polys.shp":
#            raster_layers =[#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201908300942-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201908191243-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201908020933-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201907241201-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201907101037-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201907011024-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201906250808-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201906191238-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201906031049-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201905271311-GR_clustering_output.tif",
##r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201905221036-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201905131422-GR_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201905030100_clustering_output.tif",
#r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Hendrik de Heer-201909051401-GR_clustering_output.tif"]
#            for raster_layer in raster_layers:
#                target_layer = raster_layer[:-4] + "_voronoi.tif"
#                create_plant_polygons(vector_layer, raster_layer, target_layer, p)

    p.close()
    p.join()
