import gdal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm
import cv2
from skimage import color
import numpy.ma as ma
import time
import cv2 as cv
gdal.UseExceptions()

path = r"G:\Shared drives\Archive\c07_hollandbean\Joke Visser\20190522\1245\Orthomosaic\c07_hollandbean-Joke Visser-201905221245.tif"
dst = r"D:\grass2.tif"

block_init = (25000, 30000, 5000, 5000) ##block in orthomosaic on which the KMeans is initialized
no_data_value = 3*255
sigma = 60
kmeans_init = np.array([[0,0],[-30, 30]])
blocksize = (2**12, 2**12)


#%% initiliaze clusters
ds = gdal.Open(path)
band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

array1 = band1.ReadAsArray(*block_init)
array2 = band2.ReadAsArray(*block_init)
array3 = band3.ReadAsArray(*block_init)

plt.subplot(121)
plt.imshow(np.array([array1, array2, array3]).T)

mask = np.zeros(array1.shape)
mask[array1.astype(np.uint16)+ array2.astype(np.uint16)+ array3.astype(np.uint16) == no_data_value] = 1

filtered = np.array([cv2.GaussianBlur(array1, (0, 0), sigma), cv2.GaussianBlur(array2, (0, 0), sigma), cv2.GaussianBlur(array3, (0, 0), sigma)])
lab = ma.masked_array(color.rgb2lab(filtered.T)[:,:,1:3], mask=np.array([mask, mask]).T)
array1 = None
array2 = None
array3 = None
s = lab.compressed().reshape((block_init[2]*block_init[3] - sum(sum(mask.astype(bool))), 2))

km = KMeans(2, init=kmeans_init)

t = time.time()
km.fit(s)
print(time.time() - t)
sigma60 = None
s = None
p = km.predict(lab.data.reshape((block_init[2] * block_init[3], 2))).reshape((block_init[2], block_init[3]))
lab = None
plt.subplot(122)
plt.imshow(p)

#%% create grass mask
xsize = band1.XSize
ysize = band1.YSize

mask = np.zeros((ysize, xsize)).astype(np.uint8)

offsets = []
for i in range(int(np.ceil(xsize/blocksize[0]))):
    for j in range(int(np.ceil(ysize/blocksize[1]))):
        sx = i * blocksize[0]
        ox = blocksize[0] if sx + blocksize[0] < xsize else xsize - sx
        sy = j * blocksize[1]
        oy = blocksize[1] if sy + blocksize[1] < ysize else ysize - sy
        offsets.append((sx, sy, ox, oy))
        
pbar = tqdm(total =len(offsets), desc="finding grass", position=0)
for sx, sy, ox, oy in offsets:
    r = band1.ReadAsArray(sx, sy, ox, oy)
    g = band2.ReadAsArray(sx, sy, ox, oy)
    b = band3.ReadAsArray(sx, sy, ox, oy)
    m = np.zeros(r.shape)
    m[r.astype(np.uint16)+ g.astype(np.uint16)+ b.astype(np.uint16) == no_data_value] == 1
    f = np.array([cv2.GaussianBlur(r, (0, 0), sigma), cv2.GaussianBlur(g, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)])
    lab = color.rgb2lab(f.T)[:,:,1:3].T
    v = lab.reshape((2, ox*oy)).T
    r = None
    g = None
    b = None
    
    p = km.predict(v).reshape((oy, ox))
    p[m == 1] = 0
    mask[sy:sy+oy, sx:sx+ox] = p

    pbar.update(1)
pbar.close()

#%% create tiff file
gt = ds.GetGeoTransform()
projection = ds.GetProjection()

driver = gdal.GetDriverByName('GTiff')
tiff = driver.Create(dst, mask.shape[1], mask.shape[0], 1, gdal.GDT_UInt16, options=['COMPRESS=LZW'])
tiff.SetGeoTransform(gt)
tiff.SetProjection(projection)
tiff.GetRasterBand(1).WriteArray(mask)
tiff.GetRasterBand(1).FlushCache()
tiff = None

#%% create shape file
from shapely.geometry import Polygon
contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
shapes = []

for i,c in enumerate(contours):
    if len(c) > 2 and hierarchy[0,i,3] == -1:
        shapes.append(Polygon([cc[0] for cc in c]))
        
shapes = sorted(shapes, key=lambda s : s.area, reverse=True)        
for s in shapes[:100]:
    plt.plot(*s.exterior.xy)
        
