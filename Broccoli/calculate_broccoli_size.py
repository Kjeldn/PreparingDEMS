import numpy as np
import exiftool
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import gdal
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from functools import partial

base_path = r"Z:\100 Testing\190716 Iron Man AZ74 10m difuse light\DJI_0001.JPG" ##photo of the drone taken on the ground
directory = r"Z:\100 Testing\190716 Iron Man AZ74 10m difuse light" ##directory where all the photo's are stored
ahn_path = r"C:\Users\wytze\Downloads\M_25FZ2\m_25fz2.tif" ##AHN tif file of the crop field in question
dem_path = r"Z:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190614\1505\Orthomosaic\c08_biobrass-AZ91-201906141505_DEM.tif" ##A DEM of the crop field to reproject the AHN raster

def get_value_at_coord(array, coord_x, coord_y, gt):
    return array[int(abs(np.floor((coord_x - gt[3])/gt[5])))][int(abs(np.floor((coord_y - gt[0])/gt[1])))]

def calculate_size(n_pixels_x: int, n_pixels_y: int, f: float, sensor_size: tuple, img_size: tuple, dist_from_center_x: float, dist_from_center_y: float, altitude: float):
    fov_o_w = 2* np.tan(((n_pixels_x/img_size[0]) * sensor_size[0])/(2*f))
    fov_o_t = 2* np.tan(((n_pixels_y/img_size[1]) * sensor_size[1])/(2*f))
    
    gimbal_x = np.arctan(dist_from_center_x/altitude)
    gimbal_y = np.arctan(dist_from_center_y/altitude)
    
    dist_x = altitude * (np.tan(gimbal_x + 0.5 * fov_o_w) - np.tan(gimbal_x - 0.5 * fov_o_w))
    dist_y = altitude * (np.tan(gimbal_y + 0.5 * fov_o_t) - np.tan(gimbal_y - 0.5 * fov_o_t))
    
    return (dist_x, dist_y)

def calculate_dist_from_center(n_pixels_x: int, n_pixels_y: int, img_size: tuple, f: float, altitude: float, gimbal_x: float, gimbal_y: float, sensor_size: tuple):
    s_x = 2 * (n_pixels_x/img_size[0]) * sensor_size[0]
    s_y = 2 * (n_pixels_y/img_size[1]) * sensor_size[1]
    
    fov_w = 2 * np.arctan(s_x / (2 * f))
    fov_t = 2 * np.arctan(s_y / (2 * f))
    
    d_x = altitude * (np.tan(gimbal_x + 0.5 * fov_w) - np.tan(gimbal_x - 0.5 * fov_w))
    d_y = altitude * (np.tan(gimbal_y + 0.5 * fov_t) - np.tan(gimbal_y - 0.5 * fov_t))
    return (0.5 * d_x, 0.5 * d_y)

def calculate_area_pixel(n_pixel_x: int, n_pixel_y:int, img_size: tuple, f: float, altitude: float, gimbal_x: float, gimbal_y: float, sensor_size: tuple):
    pixels_from_center_x = abs(img_size[0]/2 - n_pixel_x)
    pixels_from_center_y = abs(img_size[1]/2 - n_pixel_y)
    
    s_x = 2 * (pixels_from_center_x/img_size[0]) * sensor_size[0]
    s_y = 2 * (pixels_from_center_y/img_size[1]) * sensor_size[1]
    
    fov_w = 2 * np.arctan(s_x / (2 * f))
    fov_t = 2 * np.arctan(s_y / (2 * f))
    
    d_x = 0.5 * altitude * (np.tan(gimbal_x + 0.5 * fov_w) - np.tan(gimbal_x - 0.5 * fov_w))
    d_y = 0.5 * altitude * (np.tan(gimbal_y + 0.5 * fov_t) - np.tan(gimbal_y - 0.5 * fov_t))
    
    fov_o_w = 2* np.tan(((1/img_size[0]) * sensor_size[0])/(2*f))
    fov_o_t = 2* np.tan(((1/img_size[1]) * sensor_size[1])/(2*f))
    
    gimbal_x = np.arctan(d_x/altitude)
    gimbal_y = np.arctan(d_y/altitude)
    
    dist_x = altitude * (np.tan(gimbal_x + 0.5 * fov_o_w) - np.tan(gimbal_x - 0.5 * fov_o_w))
    dist_y = altitude * (np.tan(gimbal_y + 0.5 * fov_o_t) - np.tan(gimbal_y - 0.5 * fov_o_t))
    
    return dist_x * dist_y


paths = [f for f in listdir(directory) if isfile(join(directory, f)) and 'DJI' in f]
with exiftool.ExifTool() as et:
    md = et.get_metadata(base_path)

sensor_size = (12.8, 9.6)
coords = (md['Composite:GPSLatitude'], md['Composite:GPSLongitude'])

dem_ds = gdal.Open(dem_path)
band = dem_ds.GetRasterBand(1)
gt = dem_ds.GetGeoTransform()

with rasterio.open(ahn_path) as src:
    source = src.read(1)
    ahn_array = np.zeros((band.XSize, band.YSize))
    
    with rasterio.Env():
        reproject(
                source,
                ahn_array,
                src_transform = src.transform,
                src_crs = src.crs,
                dst_transform = A(gt[1], 0, gt[0], 0, gt[5], gt[3]),
                dst_crs = "EPSG:4326",
                respampling = Resampling.cubic
                )
            
    source = None
    
base_ahn_height = get_value_at_coord(ahn_array, coords[0], coords[1], gt)

pbar = tqdm(total=len(paths), position=0)
alts = []
for p in paths:
    with exiftool.ExifTool() as et:
        md = et.get_metadata(directory + "/" + p)
    
    rel_altitude = float(md['XMP:RelativeAltitude']) * 1000
    coords = (md['Composite:GPSLatitude'], md['Composite:GPSLongitude'])
    ahn_height = get_value_at_coord(ahn_array, coords[0], coords[1], gt)
    ahn_diff = base_ahn_height * 1000 - ahn_height * 1000
    altitude = rel_altitude - ahn_diff
    
    img_size = (md['File:ImageWidth'], md['File:ImageHeight'])
    center = (0.5 * img_size[0], 0.5 * img_size[1])
    focal_length = md['EXIF:FocalLength']
    gimbal_x = np.deg2rad(float(md['XMP:GimbalPitchDegree']) + 90)
    gimbal_y = np.deg2rad(float(md['XMP:GimbalRollDegree']))
    
    calc_area_pixel_v = np.vectorize(partial(calculate_area_pixel, img_size = img_size, f=focal_length, altitude = altitude, gimbal_x=gimbal_x, gimbal_y=gimbal_y, sensor_size=sensor_size))
    
    ##All found broccolis found in this picture: let v1 and v2 be the vectors of broccolis found of the pixel indices in x and y direction
    broccolis = []
    for b in broccolis:
        v1 = np.zeros()
        v2 = np.zeros()
        area_broccoli = sum(calc_area_pixel_v(v1, v2))
    
    pbar.update(1)
pbar.close()