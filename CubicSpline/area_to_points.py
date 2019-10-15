import fiona
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
grid_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select grid file", parent=root)
root.destroy()
areas = []
plants = geopandas.read_file(plants_path)
plants = plants.to_crs({'init': 'epsg:28992'})
plants_l = [(p.x, p.y) for p in plants.loc[:,'geometry']]                
spindex = Index(Polygon(plants_l).bounds)
for i, plant in enumerate(plants_l):
    spindex.insert({'index': i, 'plant': plant}, (plant[0], plant[1], plant[0], plant[1]))
         
keys = []
pbar = tqdm(total = len(polys_paths), desc="Computing areas", position=0)
for path in polys_paths:
    polys = geopandas.read_file(path)
                
    area_string = re.findall(r'\d+', path)[-1]
    keys.append('area' + area_string)
    areas_i = []
    for poly in polys.loc[:,'geometry']:
        if poly.is_valid:
            plants_i = spindex.intersect(poly.bounds)
            if isinstance(plants_i, list):
                n = sum([poly.contains(Point(plant['plant'])) for plant in plants_i])
                for plant in plants_i:
                    if Point(plant['plant']).within(poly):
                        plant['area' + area_string] = (poly.area / n)
                        areas_i.append((poly.area / n))
            else:
                n = 1
                if Point(plants_i['plant']).within(poly):
                    plants_i['area'] = (poly.area / n)
                    areas_i.append((poly.area / n))
    areas.append(areas_i)
    pbar.update(1)
pbar.close()

grid = geopandas.read_file(grid_path)
grid = grid.to_crs({'init': 'epsg:28992'})
        
for i in range(len(grid.loc[:,'geometry'])):
    poly = grid.loc[i,'geometry']
    total_area = {}
    n = 0
    for plant in spindex.intersect(poly.bounds):
        if Point(plant['plant']).within(poly):
            n+=1
            for key in keys:
                if key in plant:
                    if key in total_area:
                        total_area[key] += plant[key]
                    else:
                        total_area[key] = plant[key]
    for key in keys:
        if key in total_area:
            grid.loc[i, key] = total_area[key]/n

grid.to_file("/".join(plants_path.split("/")[:-1]) + "/areas.shp")

    
