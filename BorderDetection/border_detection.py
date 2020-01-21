import gdal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm
gdal.UseExceptions()

path = r"G:\Gedeelde drives\Archive\c01_verdonk\Wever oost\20190717\0731\Orthomosaic\c01_verdonk-Wever oost-20190717.tif"

ds = gdal.Open(path)
band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

array1 = band1.ReadAsArray(20000, 20000, 5000, 5000)
array2 = band2.ReadAsArray(20000, 20000, 5000, 5000)
array3 = band3.ReadAsArray(20000, 20000, 5000, 5000)

#array4 = 2 * array2 - array1 - array3
#ret, thresh1 = cv2.threshold(array4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#f1 = cv2.blur(f1, (101, 101))
#f2 = cv2.blur(f2, (101, 101))
#f3 = cv2.blur(f3, (101, 101))
#
#f4 = 2 * f2 - f1 - f3
#ret2, thresh2 = cv2.threshold(array4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##ret2, thresh2 = cv2.threshold(f4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#f5 = cv2.blur(thresh2, (51,51))
#plt.imshow(f5)

#img2 = np.array([array1, array2, array3])


sigma60 = np.array([gaussian_filter(array1, 60), gaussian_filter(array2, 60), gaussian_filter(array3, 60)]).reshape((3, 5000**2))
array1 = None
array2 = None
array3 = None

a = KMeans(3)

a.fit(sigma60.T)
sigma60 = None
green_label = np.argmax([r[1] - np.mean([r[0], r[2]]) for r in a.cluster_centers_])

xsize = band1.XSize
ysize = band1.YSize

mask = np.zeros((xsize, ysize)).astype(bool)


#offset_x = int(xsize/3)
#offset_y = int(ysize/3)
offset_x = 5120
offset_y = 5120
offsets = []
for i in range(int(np.ceil(xsize/offset_x))):
    for j in range(int(np.ceil(ysize/offset_y))):
        sx = i * offset_x
        ox = offset_x if sx + offset_x < xsize else xsize - sx
        sy = j * offset_y
        oy = offset_y if sy + offset_y < ysize else ysize - sy
        offsets.append((sx, sy, ox, oy))
     
        
pbar = tqdm(total =len(offsets), desc="finding grass", position=0)
for offset in offsets:

    r1 = band1.ReadAsArray(*offset)
    g1 = band2.ReadAsArray(*offset)
    b1 = band3.ReadAsArray(*offset)
    
    i1 = np.array([gaussian_filter(r1, 60), gaussian_filter(g1, 60), gaussian_filter(b1, 60)]).reshape((3, offset_x * offset_y))
    r1 = None
    g1 = None
    b1 = None
    
    p1 = a.predict(i1.T).T.reshape((offset_x, offset_y))
    p1[p1 == green_label] = 255
    p1[p1 < 255] = 0
    mask[offset[0]:offset[0]+offset[2], offset[1]:offset[1]+offset[3]] = p1
    pbar.update(1)
pbar.close()

#gt = ds.GetGeoTransform()
#projection = ds.GetProjection()
#
#driver = gdal.GetDriverByName('GTiff')
#tiff = driver.Create(dest, f1.shape[1], f1.shape[0], 1, f1.GDT_Float32, options=['COMPRESS=LZW'])
#tiff.SetGeoTransform(gt)
#tiff.SetProjection(projection)
#tiff.GetRasterBand(1).WriteArray(f1)
#tiff.GetRasterBand(1).FlushCache()
#tiff.GetRasterBand(2).WriteArray(f2)
#tiff.GetRasterBand(2).FlushCache()
#tiff.GetRasterBand(3).WriteArray(f3)
#tiff.GetRasterBand(3).FlushCache()
#tiff = None