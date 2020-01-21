"""
Reads plant count file, creates voronoi diagram and writes voronoi polygons to shape file.
Location of written shape file is "_".join(plant_path.split("_")[:-1] + "_voronoi_polys.shp")
CRS is epsg:4326.
"""

import geopandas
from scipy.spatial import Voronoi
import util_voronoi as utilv
from shapely.geometry import Polygon
import numpy as np
#from tkinter import filedialog
#from tkinter import *

#root = Tk()
#plant_path = filedialog.askopenfilename(initialdir =  r"D:", title="Select plant count", parent=root)
#plant_path= r"D:\VanBovenDrive\VanBoven MT\Archive\c01_verdonk\Wever west\20190717\0749\Plant_count\20190717_count.shp"
#root.destroy()
def create_voronoi_from_points(plant_path):
    output_name = plant_path[:-4] + "_voronoi_polys.shp"
    plants = geopandas.read_file(plant_path)
    plants = plants.mask(plants.geometry.eq('None')).dropna()
    plants_r, mx, my = utilv.readable_values(np.array([(p.centroid.xy[0][0], p.centroid.xy[1][0]) for p in plants.loc[:,'geometry']]))
    vor = Voronoi(plants_r)
    convex_hull = Polygon([(p.centroid.xy[0][0], p.centroid.xy[1][0]) for p in plants.loc[:,'geometry']]).convex_hull
    vor_polys = []
    for r in vor.regions:
        if -1 not in r and len(r) > 2 and Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)).within(convex_hull):
            vor_polys.append(Polygon(utilv.readable_values_inv(vor.vertices[r], mx, my)))   
            
    df = geopandas.GeoDataFrame({'geometry': geopandas.GeoSeries(vor_polys), 'index': np.arange(1, len(vor_polys) + 1)})
    df.crs = {'init': 'epsg:4326'}
    df.to_file(output_name)
    return output_name

if __name__ == "__main__":
    create_voronoi_from_points(r"D:\800 Operational\c07_hollandbean\Season evaluation\Mellisant\POINTS.shp")
