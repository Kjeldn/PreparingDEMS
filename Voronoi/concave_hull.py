import fiona
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import numpy as np

path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\20190717_count.shp"
dst = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli\20190717_count_missed.shp"

def get_convex_hull(plants):
    poly = Polygon(zip(plants[:,0], plants[:,1]))
    poly_line = LinearRing(np.array([z.tolist() for z in poly.convex_hull.exterior.coords.xy]).T)
    polygon = Polygon(poly_line.coords)
    return polygon

with fiona.open(path) as src:
    plants = []
    src_driver = src.driver
    src_crs = src.crs
    src_schema = src.schema
    for i in range(len(src)):
        if src[i]['geometry']:
            plants.append([src[i]['geometry']['coordinates'][0][0], src[i]['geometry']['coordinates'][0][1]])
            
plants = sorted(sorted(plants, key=lambda a : a[0]), key = lambda a: a[1])
convex_hull = get_convex_hull(np.array(plants))


