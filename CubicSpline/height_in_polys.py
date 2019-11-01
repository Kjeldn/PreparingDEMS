import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from tqdm import tqdm
from tkinter import filedialog
from tkinter import *

root = Tk()
polys_path = filedialog.askopenfilename(initialdir =  r"Z:\800 Operational\c07_hollandbean\Joke visser", title="Select polys shapefile", parent=root)
dem_path = filedialog.askopenfilename(initialdir =  r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\Plants_compare\Joke Visser", title="Select dem tif", parent=root)
root.destroy()

shp = gpd.read_file(polys_path)
shp = shp.to_crs({'init': 'epsg:4326'})

pbar = tqdm(total=len(shp.loc[:,'geometry']), desc="getting mean height", position=0)

with rasterio.open(dem_path) as src:
    for i in range(len(shp.loc[:,'geometry'])):
        if shp.loc[i,'geometry']:
            out_image, out_transform = mask(src, [mapping(shp.loc[i,'geometry'])], crop=True)
            elev = np.extract(out_image[0,:,:] != 0, out_image[0,:,:])
            shp.loc[i, 'mean_heigh'] = np.mean(sorted(elev, reverse=True)[:int(len(elev)/2)])
        pbar.update(1)

shp = shp.to_crs({'init': 'epsg:28992'})
shp.to_file(polys_path)