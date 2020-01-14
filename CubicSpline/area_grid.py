# -*- coding: utf-8 -*-
#This script creates a csv and a gpkg file containing the area of vegetation in one part of a grid
import geopandas as gpd
import numpy as np
import os

#Paths to the shapefiles created by the rasterize_voronoi script. The order of the links is not important.
paths_list = [
r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links\c03_termote-Binnendijk Links-201906101608-GR_clustering_voronoi.shp",
r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links\c03_termote-Binnendijk Links-201906161329-GR_clustering_voronoi.shp",
r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links\c03_termote-Binnendijk Links-201906251610-GR_clustering_voronoi.shp",
r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links\c03_termote-Binnendijk Links-201905080000_clustering_voronoi.shp"
]

#Grid path, the grid needs to be in epsg:4326
path_grid = r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links\grid\clip_shp_empty_grid.shp"
#Output path of the csv file and the geopackage
path_output = r"D:\800 Operational\c03_termote\Season Evaluation\Binnendijk Links"

length_grid = len(gpd.read_file(paths_list[0]))

x = np.full(length_grid, np.nan)
y = np.full(length_grid, np.nan)
geometry = gpd.read_file(path_grid)['geometry']

table = gpd.GeoDataFrame(list(zip(x,y,geometry)), columns = ['x','y','geometry'])

#Adds the area per date to the table
datecodelist = []
for path in paths_list:
    filename = (os.path.split(path)[1])
    datecode = filename.split('-')[2][0:12]
    area_datecode = 'area'+datecode
    datecodelist.append(datecode)

    #Changes the coordinate system to calculate the correct area
    shp_data_wsg84 = gpd.read_file(path)
    shp_data_rdnew = shp_data_wsg84.to_crs({'init': 'epsg:28992'})

    area_grid = shp_data_rdnew.area
    table[area_datecode] = area_grid

#Sort the colums of the table in chronological order
sorted_colums = sorted(datecodelist)
sorted_area_datecodes = ['x','y']

for sorted_colum in sorted_colums:
    sorted_area_datecodes.append(str('area'+sorted_colum))
    
#Create csv table
table_csv = table[sorted_area_datecodes]

#Export csv file
path_output_csv = path_output+r"\area_grid.csv"
table_csv.to_csv(path_output_csv)

#Create gpkg table
sorted_area_datecodes.append('geometry')
table_gpkg = table[sorted_area_datecodes]
table_gpkg.crs = {'init': 'epsg:4326'}
#Checks if the projection of the grid is in epsg:4326
if gpd.read_file(path_grid)['geometry'].crs != {'init': 'epsg:4326'}:
    print('Grid input is not in epsg:4326 so the output of .gpkg file is also not epsg:4326!!')

#Export gpkg file, geometry is in wgs84 projection
path_output_gpkg = path_output+r"\area_grid.gpkg"
table_gpkg.to_file(path_output_gpkg, driver="GPKG")
