import gdal
import numpy as np
from typing import List
from scipy import interpolate
import rasterio
from rasterio.warp import reproject, Resampling

path_original = r"C:/Users/wytze/20190603_modified.tif"
path_ahn = r"C:\Users\wytze\OneDrive\Documents\vanBoven\190723_Demo_Hendrik_de_Heer\DEM\m_43ez2.tif"
path_ridges =  r"C:/Users/wytze/crop_top.tif"
path_destination = r"C:\Users\wytze\OneDrive\Documents\vanBoven\190723_Demo_Hendrik_de_Heer\DEM\cubic_spline.tif"

vertices = [(51.73861232, 4.38036677), (51.73829906, 4.38376774), (51.73627495, 4.38329311), (51.73661151, 4.37993097)]

#%% Polygon and Point class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Polygon:
    EXTREME = 100000000
    def __init__(self, vs: List[Point]):
        self.vs = vs
    
    """Given three colinear points p, q, r, the function checks if q lies on line segment pr
    
    :param p: start of line segment pr
    :param q: the point to check
    :param r: end of line segment pr
    :returns: True if q on line segment pr
    """
    def on_segment(self, p, q, r):
        if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
            return True
        return False
    
    """Find orientation of ordered triplet (p, q, r)
    0 --> p, q, r are colinear
    1 --> Clockwise
    2 --> Counter clockwise
    
    :param p, q, r: points of which the orientation is checked
    :returns: orientation of the given points
    """
    def orientation(self, p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if (val == 0): 
            return 0
        return 1 if val > 0 else 2
    
    """Check if line segment p1q1 and p2q2 intersect
    
    :param p1, q1, p1, p2: the points of the line segments which are checked for intersection
    :returns: True if p1q1 and p2q2 intersect
    """
    def do_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
        
        if (o1 != o2 and o3 != o4):
            return True
        
        if (o1 == 0 and self.on_segment(p1, p2, q1)):
            return True
        
        if (o2 == 0 and self.on_segment(p1, q2, q1)):
            return True
        
        if (o3 == 0 and self.on_segment(p2, p1, q2)):
            return True
        
        if (o4 == 0 and self.on_segment(p2, q1, q2)):
            return True
        
        return False
    
    """Check if point p is inside the n-sized polygon given by points vs
    
    :param vs: list of points which make up the polygon
    :param n: number of vertices of the polygon
    :param p: the point to be checked
    :returns: True if p lies inside the polygon
    """
    def is_inside_polygon(self, p: Point):
        if len(self.vs) < 3:
            return False
        
        extreme = Point(self.EXTREME, p.y)
        
        count = 0
        i = 0
        while True:
            next_i = (i + 1) % len(self.vs)
            
            if self.do_intersect(self.vs[i], self.vs[next_i], p, extreme):
                if self.orientation(self.vs[i], p, self.vs[next_i]) == 0:
                    return self.on_segment(self.vs[i], p, self.vs[next_i])
                count += 1
                
            i = next_i
            if i == 0:
                break
        
        return count % 2 == 1

    
def create_tiff(array, gt, projection, dest: str):
    driver = gdal.GetDriverByName('GTiff')
    tiff = driver.Create(dest, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    tiff.SetGeoTransform(gt)
    tiff.SetProjection(projection)
    tiff.GetRasterBand(1).WriteArray(array)
    tiff.GetRasterBand(1).FlushCache()
    tiff = None
    

#%% reproject AHN Model
with rasterio.open(path_original) as dst:
    dst_shape = (dst.height, dst.width)
    dst_transform = dst.transform
    dst_crs = dst.crs
    ahn_array = np.zeros(dst_shape)
    
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


#%% cubic spline
file = gdal.Open(path_original)
ridges = gdal.Open(path_ridges)

ridges_array = ridges.GetRasterBand(1).ReadAsArray()

band = file.GetRasterBand(1)
array = band.ReadAsArray()
projection = file.GetProjection()
gt = file.GetGeoTransform()
xsize = band.XSize
ysize = band.YSize

##Remove all non-values from array
array[array < 0] = 0

vs_i = []

for i in range(len(vertices)):
    vs_i.append(Point(int(abs(np.floor((vertices[i][0] - gt[3])/gt[5]))), int(abs(np.floor((vertices[i][1] - gt[0])/gt[1])))))

poly = Polygon(vs_i)

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
        if array[step * i, step * j] == 0 or ridges_array[step * i, step * j] == -1 or abs(ahn_array[step * i, step * j]) > 2 or not poly.is_inside_polygon(Point(step * i, step * j)):
            mask[i][j] = True

z = np.ma.array(data, mask=mask)

##Remove all points which are either a non-value, not bare ground, non-value in AHN or not in the polygon
z1 = z[~z.mask]
y1 = y[~z.mask]
x1 = x[~z.mask]

xnew, ynew = np.mgrid[0:ysize, 0:xsize]

tck = interpolate.bisplrep(x1, y1, z1, s=len(z) - np.sqrt(2*len(z)))
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

create_tiff(array - znew, gt, projection, path_destination)