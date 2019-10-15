import fiona
from shapely.geometry import shape, Polygon, Point, MultiPolygon, mapping
from pyqtree import Index
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import util_voronoi as util
from tqdm import tqdm

plant_path = r"Z:\800 Operational\c07_hollandbean\Joke visser\Count\20190603_final_plant_count _inside.gpkg"
polys_path = r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906191208-GR_clustering_polys2.shp"

plants = []
with fiona.open(plant_path) as src:
    for s in src:
        if s['geometry']['type'] == 'Point':
            plants.append(s['geometry']['coordinates'])
        elif s['geometry']['type'] == 'MultiPoint':
            plants.append(s['geometry']['coordinates'][0])
plants = np.array(plants)
            
polys = []
with fiona.open(polys_path) as src:
    for s in src:
        if s['geometry']:
            polys.append(shape(s['geometry']))
            
spindex = Index((np.amin(plants[:, 0]), np.amin(plants[:,1]), np.amax(plants[:, 0]), np.amax(plants[:,1])))
for p in plants:
    spindex.insert(p, (p[0], p[1], p[0], p[1]))
    
for j in range(len(polys)):
    polys[j] = polys[j].buffer(0)
    
#%%
multi_ints = []
pbar1 = tqdm(total=len(polys), desc="finding intersections", position=0)
for i in range(len(polys)):
    pbar1.update(1)
    if not polys[i].bounds:
        continue
    bounds = polys[i].bounds
    offset = min(abs(bounds[0] - bounds[2]), abs(bounds[1] - bounds[3]))
    plants_i = spindex.intersect((bounds[0] - offset, bounds[1] - offset, bounds[2] + offset, bounds[3] + offset))
    if plants_i and len(plants_i) > 3:
        plants_r, mean_x_coord, mean_y_coord = util.readable_values(np.array(plants_i))
        vor = Voronoi(plants_r)
        
        vor_polys = []
        for p in vor.point_region:
            r = vor.regions[p]
            if -1 not in r and len(r) > 2:
                vor_polys.append(Polygon(util.readable_values_inv(vor.vertices[r], mean_x_coord, mean_y_coord)))
          
        ints = []
        areas = []
        for vor_poly in vor_polys:
            for poly in polys[i]:
                intersect = vor_poly.intersection(poly)
                if not intersect.is_empty:
                    if intersect.type == "Polygon":
                        ints.append(intersect)
                    elif intersect.type == "MultiPolygon":
                        for polyy in intersect:
                            ints.append(polyy)
                    areas.append(intersect.area)
        multi_ints.append(MultiPolygon(ints))
pbar1.close()        
        
pbar2 = tqdm(total=len(multi_ints), desc="write shape file", position=0)
with fiona.open(r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\voronoi_polys_hybrid.shp", 'w', crs={'init': 'epsg:4326'}, driver='ESRI Shapefile', schema={'geometry': 'MultiPolygon', 'properties': {'index': 'int'}}) as c:
    for i in range(len(multi_ints)):
        c.write({
            'geometry': mapping(multi_ints[i]),
            'properties': { 'index': i }
        })
        pbar2.update(1)

#%%
polys_paths = [r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906031020_clustering_polys.shp"]
# =============================================================================
#                 r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906191208-GR_clustering_polys.shp",
#                 r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201906250739-GR_clustering_polys.shp",
#                 r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201907010933-GR_clustering_polys.shp",
#                 r"Z:\800 Operational\c07_hollandbean\Joke visser\Plant_area\c07_hollandbean-Joke Visser-201907101007-GR_clustering_polys.shp"]
# =============================================================================

grid_path = r"Z:\800 Operational\c07_hollandbean\Joke visser\empty_grid.gpkg"

grid = []
with fiona.open(grid_path) as src:
    for s in src:
        if s['geometry']:
            grid.append(shape(s['geometry']))

areas = [[] for i in range(len(polys_paths))]

for path in polys_paths:
    polys = [[] for i in range(len(grid))]
    with fiona.open(polys_path) as src:
        for s in src:
            if s['geometry']:
                polys[s['properties']['index']].append(shape(s['geometry']))
                
    
