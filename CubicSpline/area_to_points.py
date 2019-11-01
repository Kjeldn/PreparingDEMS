"""
This script calculates the area of the polygon a Point in a plant count shape file divided by the number of 
plants in the polygon. It does this for all polygon files loaded in polys_paths. And adds columns to the dataframe
describing the plant count for the areas of all polygon files.
"""
from shapely.geometry import Point, MultiPolygon, Polygon, shape, mapping
from pyqtree import Index
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from tqdm import tqdm
import re

root = Tk()
polys_paths = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select polys file", parent=root, multiple=True)
plants_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select plant file", parent=root)
root.destroy()
plants = geopandas.read_file(plants_path)
plants = plants.to_crs({'init': 'epsg:28992'})
            
spindex = Index(Polygon([(p.x, p.y) for p in plants.loc[:,'geometry']]).bounds)
for i in plants.index:
    spindex.insert({
            'index': i, 
            'plant': list(plants.loc[i,'geometry'].coords)[0]
            }, 
    (list(plants.loc[i,'geometry'].coords)[0][0], 
     list(plants.loc[i,'geometry'].coords)[0][1], 
     list(plants.loc[i,'geometry'].coords)[0][0], 
     list(plants.loc[i,'geometry'].coords)[0][1]))
         
pbar = tqdm(total = len(polys_paths), desc="Computing areas", position=0)
for path in polys_paths:
    polys = geopandas.read_file(path)
    polys = polys.to_crs({'init': 'epsg:28992'})
    key = 'area'+ re.findall(r'\d+', path)[-1]
    pbar2 = tqdm(total = len(polys.index), desc="computing areas", position=0)
    for poly in polys.loc[:,'geometry']:
        if poly.is_valid:
            plants_i = spindex.intersect(poly.bounds)
            if isinstance(plants_i, list):
                n = sum([poly.contains(Point(plant['plant'])) for plant in plants_i])
                for plant in plants_i:
                    if Point(plant['plant']).within(poly):
                        plants.loc[plant['index'], key] = (poly.area / n)
            else:
                n = 1
                if Point(plants_i['plant']).within(poly):
                    plants.loc[plant['index'], key] = (poly.area / n)
        pbar2.update(1)
    pbar2.close()
    pbar.update(1)
pbar.close()

#plants.to_file("/".join(plants_path.split("/")[:-1]) + "/areas.shp")
from matplotlib.collections import LineCollection

segs = []
n= 50000
pbar3 = tqdm(total =n, desc="plotting...", position=0)
keys = list(filter(lambda k : 'area2' in k, plants.keys()))
x= np.arange(len(keys))
segs = np.zeros((n, len(x), 2))
y = np.array([[plants.loc[0,k] for k in keys],[plants.loc[1,k] for k in keys]])
for i in plants.index[2:50000]:
    y = np.vstack((y,np.array([plants.loc[i,k] for k in keys])))
    pbar3.update(1)
pbar3.close()

segs[:,:, 1] = y
segs[:,:,0] = x

fig, ax = plt.subplots()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 0.4)

ax.add_collection(LineCollection(segs, linewidths=(0.5, 1, 1.5, 2), linestyle="solid", colors="blue", alpha=0.004))
plt.show()
