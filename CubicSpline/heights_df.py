import geopandas as gpd

from pyqtree import Index
from shapely.geometry import mapping, MultiPoint, Point, Polygon
from shapely.strtree import STRtree
import re
import csv
from tqdm import tqdm
import scipy
import numpy as np

csv_dest = r"D:\800 Operational\c07_hollandbean\Season evaluation\Joke Visser\heights.csv"

voronoi_paths = [r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908300729-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201909060802-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905221245_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905271514-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906031020-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906191208-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906250739-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907010933-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907101007-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908020829-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908231004-GR_clustering_voronoi.shp"]
plantcount_path = r"D:\800 Operational\c07_hollandbean\Season evaluation\Counts\joke_shp_test.shp"

df = gpd.read_file(plantcount_path)
for i in df.index:
    if not df.loc[i,'geometry']:
        df = df.drop(i)
df = df.to_crs({'init': 'epsg:28992'})
bbox = MultiPoint([df.loc[i,'geometry'][0] if isinstance(df.loc[i,'geometry'], MultiPoint) else df.loc[i,'geometry'] for i in df.index]).bounds
keys = []
plants = df.loc[:, 'geometry']
polys_list = []
heights_list = []
heights = []
pbar0 = tqdm(total=len(voronoi_paths), desc="reading voronoi polygons", position=0)
for p in voronoi_paths:
    df['height' + re.findall(r'\d+', p)[-1]] = None
    keys.append('height' + re.findall(r'\d+', p)[-1])
    f = gpd.read_file(p)
    f = f.to_crs({'init': 'epsg:28992'})
    polys_list.append(f.loc[:, 'geometry'])
    heights_list.append(f.loc[:, 'mean_heigh'])
    heights.append([0 for _ in range(len(plants))])
    pbar0.update(1)
pbar0.close()
    
pyq = Index(bbox=bbox)
for i, p  in enumerate(plants):
    pyq.insert(i, p.bounds)
    
convex_hull = Polygon([(p.x, p.y) for p in plants]).convex_hull

pbar = tqdm(total=len(voronoi_paths), position=0)
for j, f in enumerate(polys_list):
    pbar1 = tqdm(total=len(polys_list[j]), position=0)
    for i,p in enumerate(f):
        if (p.is_valid and p.within(convex_hull)) or (not p.is_valid and p.buffer(0).within(convex_hull)):
            bbox = p.bounds
            intersected = pyq.intersect(bbox)
            if intersected:
                for k in intersected:
                    if not np.isnan(heights_list[j][i]) and abs(heights_list[j][i]) > 0 and abs(heights_list[j][i]) < 10:
                        heights[j][k] = heights_list[j][i] / len(intersected)
        pbar1.update(1)
    pbar.update(1)
    pbar1.close()
pbar.close()

with open(csv_dest, "w") as file:
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['x','y'] + keys)
    for i in range(len(plants)):
        p = plants[i] if isinstance(plants[i], Point) else plants[i][0]
        writer.writerow([p.x, p.y] + [heights[j][i] for j in range(len(keys))])
        
#%%
import copy
import matplotlib.pyplot as plt
means = []
for a in heights:
    ac = copy.deepcopy(a)
    m = np.mean(ac)
    std2 = 2*np.std(ac)
    for i in range(len(ac) -1, -1, -1):
        if abs(ac[i] - m) > std2:
            del ac[i]
    means.append(np.mean(ac))
    
plt.plot(means)