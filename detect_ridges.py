import numpy as np
import cv2
import gdal

output_path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Boomgaard\c03_termote-De Boomgaard-201907110000_DEM_crop_top.tif"
src_path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Boomgaard\c03_termote-De Boomgaard-201907110000_DEM.tif"
out_path = r'C:\Users\wytze\OneDrive\Documents\vanBoven\Boomgaard\test_resampeld.tif'

ds = gdal.Open(src_path)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

dem = ds.GetRasterBand(1).ReadAsArray()
gt = ds.GetGeoTransform()
projection = ds.GetProjection()
dem[dem == np.amin(dem)] = 0
dem[dem > 10] = 0

kernel = np.ones((8,1),np.float32)/8
filtered_dem = cv2.filter2D(dem,-1,kernel)
for i in range(2):
    filtered_dem = cv2.filter2D(filtered_dem,-1,kernel)

n=9
kernel = np.ones((n,n),np.float32)/(n**2)
smooth = cv2.filter2D(dem,-1,kernel)

ridges = (filtered_dem-smooth)
ridges[ridges > 0] = 255
ridges = ridges.astype(np.uint8)

driver = gdal.GetDriverByName('GTiff')
tiff = driver.Create(output_path, ridges.shape[1], ridges.shape[0], 1, gdal.GDT_Int16)
tiff.SetGeoTransform(gt)
tiff.SetProjection(projection)
tiff.GetRasterBand(1).WriteArray(ridges)
tiff.GetRasterBand(1).FlushCache()
tiff = None