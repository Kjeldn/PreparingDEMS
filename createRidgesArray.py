# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:04:46 2019

@author: wytze
"""

    """ 
a=1
wt = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Broccoli"
path_original = ["c01_verdonk-Wever oost-201908041528_DEM-GR", "c01_verdonk-Wever oost-201907240707_DEM-GR", "c01_verdonk-Wever oost-201907170731_DEM-GR"]
path_ahn = None #"m_19fn2.tif"
use_ridges = True

vertices = [(52.68228641, 5.12111562), (52.68255410, 5.12205255), (52.67776578, 5.12602615), (52.67753147, 5.12499776)]

file = gdal.Open(wt + "/" + path_original[a] + ".tif")
    
band = file.GetRasterBand(1)
array = band.ReadAsArray()
projection = file.GetProjection()
gt = file.GetGeoTransform()
xsize = band.XSize
ysize = band.YSize

vs_i = []

for i in range(len(vertices)):
    vs_i.append(util.Point(int(abs(np.floor((vertices[i][0] - gt[3])/gt[5]))), int(abs(np.floor((vertices[i][1] - gt[0])/gt[1])))))

poly = util.Polygon(vs_i)

xmax = 0
ymax = 0
xmin = xsize
ymin = ysize

for p in poly.vs:
    xmax = p.x if p.x > xmax else xmax
    ymax = p.y if p.y > ymax else ymax
    xmin = p.x if p.x < xmin else xmin
    ymin = p.y if p.y < ymin else ymin
    
if use_ridges: 
    ridges_array = get_ridges_array(array[ymin:ymax, xmin:xmax])
    
temp = np.zeros((ysize, xsize))
temp[ymin:ymax, xmin:xmax] = ridges_array
ridges_array = temp
del temp
util.create_tiff(ridges_array, gt, projection, wt + "/" + path_original[a] +'_ridges.tif', gdal.GDT_Int16)
"""