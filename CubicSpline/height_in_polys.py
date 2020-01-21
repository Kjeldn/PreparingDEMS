import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from tqdm import tqdm

polys_paths = [r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201909060802-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905221245_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201905271514-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906031020-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906191208-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201906250739-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907010933-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201907101007-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908020829-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908231004-GR_clustering_voronoi.shp",
r"D:\800 Operational\c07_hollandbean\Season evaluation\c07_hollandbean-Joke Visser-201908300729-GR_clustering_voronoi.shp"]

dem_paths = [
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190906\0802\Orthomosaic\c07_hollandbean-Joke Visser-201909060802_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190522\1245\Orthomosaic\c07_hollandbean-Joke Visser-201905221245_DEM_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190527\1514\Orthomosaic\c07_hollandbean-Joke Visser-201905271514_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190603\1020\Orthomosaic\c07_hollandbean-Joke Visser-201906031020_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190619\1208\Orthomosaic\c07_hollandbean-Joke Visser-201906191208_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190625\0739\Orthomosaic\c07_hollandbean-Joke Visser-201906250739_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190701\0933\Orthomosaic\c07_hollandbean-Joke Visser-201907010933_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190710\1007\Orthomosaic\c07_hollandbean-Joke Visser-201907101007_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190802\0829\Orthomosaic\c07_hollandbean-Joke Visser-201908020829_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190823\1004\Orthomosaic\c07_hollandbean-Joke Visser-201908231004_DEM-GR_cubic.tif",
r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190830\0729\Orthomosaic\c07_hollandbean-Joke Visser-201908300729_DEM-GR_cubic.tif"
]

for j in range(len(polys_paths)):
    shp = gpd.read_file(polys_paths[j])
    shp = shp.to_crs({'init': 'epsg:4326'})
    
    pbar = tqdm(total=len(shp.loc[:,'geometry']), desc="getting mean height", position=0)
    
    with rasterio.open(dem_paths[j]) as src:
        for i in range(len(shp.loc[:,'geometry'])):
            if shp.loc[i,'geometry']:
                out_image, out_transform = mask(src, [mapping(shp.loc[i,'geometry'])], crop=True)
                elev = np.extract(out_image[0,:,:] != 0, out_image[0,:,:])
                shp.loc[i, 'mean_heigh'] = np.mean(sorted(elev, reverse=True)[:int(len(elev)/2)])
            pbar.update(1)
    
    shp = shp.to_crs({'init': 'epsg:28992'})
    shp.to_file(polys_paths[j])