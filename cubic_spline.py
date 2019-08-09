import gdal
import numpy as np
from scipy import interpolate
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import Affine as A
import detect_ridges as dt
import util

path_original = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Tulips\DEM\Achter_de_rolkas-20190420-DEM_full_plot.tif"
path_ahn = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Tulips\DEM\m_24hz2.tif"
path_destination = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Tulips\DEM\Achter_de_rolkas-20190420-DEM_full_plot_cubic_spline.tif"
use_ridges = True

vertices = [(52.28058744, 4.53064423), (52.28007165, 4.53009194), (52.27968238, 4.53250545), (52.27956559, 4.53241543), (52.27864350, 4.53447008), (52.27810824, 4.53390806)]

#%% reproject AHN Model
orig = gdal.Open(path_original)
dst_shape = (orig.GetRasterBand(1).YSize, orig.GetRasterBand(1).XSize)
dst_transform = A(orig.GetGeoTransform()[1], 0, orig.GetGeoTransform()[0], 0, orig.GetGeoTransform()[5],  orig.GetGeoTransform()[3])
ahn_array = np.zeros(dst_shape)
dst_crs = "EPSG:4326"

with rasterio.open(path_ahn) as src:
    source = src.read(1)
    
    with rasterio.Env():
        reproject(
                source,
                ahn_array,
                src_transform = src.transform,
                src_crs = src.crs,
                dst_transform = dst_transform,
                dst_crs = dst_crs,
                respampling = Resampling.nearest
                )
        
        ##util.create_tiff(ahn_array, orig.GetGeoTransform(), orig.GetProjection(), 'ahn.tif')

#%% cubic spline
file = gdal.Open(path_original)
if use_ridges: 
    ridges_array, _, _ = dt.get_ridges_array(path_original)

band = file.GetRasterBand(1)
array = band.ReadAsArray()
projection = file.GetProjection()
gt = file.GetGeoTransform()
xsize = band.XSize
ysize = band.YSize

##Remove all non-values from array
array[array == np.amin(array)] = 0

vs_i = []

for i in range(len(vertices)):
    vs_i.append(util.Point(int(abs(np.floor((vertices[i][0] - gt[3])/gt[5]))), int(abs(np.floor((vertices[i][1] - gt[0])/gt[1])))))

poly = util.Polygon(vs_i)

##The space between possible bare ground points to fit over
step = 40

data = np.zeros((int(ysize/step), int(xsize/step)))
mask = np.zeros((int(ysize/step), int(xsize/step))) > 0
x = np.zeros((int(ysize/step), int(xsize/step)))
y = np.zeros((int(ysize/step), int(xsize/step)))
    
# create list of points inside the field to get the fit over
for i in range(int(ysize/step)):
    for j in range(int(xsize/step)):
        data[i][j] = array[step * i, step * j] - ahn_array[step * i, step * j]
        x[i][j] = step * i
        y[i][j] = step * j
        if array[step * i, step * j] == 0 or abs(ahn_array[step * i, step * j]) > 10 or not poly.is_inside_polygon(util.Point(step * i, step * j)):
            mask[i][j] = True
            if use_ridges and ridges_array[step * i, step * j] == 0:
                mask[i][j] = True

z = np.ma.array(data, mask=mask)

##Remove all points which are either a non-value, not bare ground, non-value in AHN or not in the polygon
z1 = z[~z.mask]
y1 = y[~z.mask]
x1 = x[~z.mask]

xnew, ynew = np.mgrid[0:ysize, 0:xsize]
tck, fp, ier, msg = interpolate.bisplrep(x1, y1, z1, full_output = 1)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

util.create_tiff(array - znew, gt, projection, path_destination)