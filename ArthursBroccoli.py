import META
import gdal
import cv2
import numpy as np
import numpy.matlib
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from tqdm import tqdm
# 0 1 2 22 26 27 30 34 43 50 56
path = r"D:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\ORTHODUMP\Arthur - Broccoli\broc_0.png"

# Arthur's Broccoli
file                               = gdal.Open(path)
gt                                 = file.GetGeoTransform()
R                                  = file.GetRasterBand(1).ReadAsArray()
G                                  = file.GetRasterBand(2).ReadAsArray()
B                                  = file.GetRasterBand(3).ReadAsArray()
img                              = np.zeros([B.shape[0],B.shape[1],3], np.uint8)
img[:,:,0]                       = B
img[:,:,1]                       = G
img[:,:,2]                       = R
img_cielab                       = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L                                  = img_cielab[:,:,0] 
hist                               = np.histogram(L,bins=256)[0]
cdf                                = hist.cumsum()
cdf_m                              = np.ma.masked_equal(cdf,0)
cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
L_eq                               = cdf[L] 

img_cielab_eq                    = img_cielab.copy()
img_cielab_eq[:,:,0]             = L_eq   
img_eq                           = cv2.cvtColor(img_cielab_eq, cv2.COLOR_Lab2BGR)
img_g                              = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)


img_b                              = cv2.bilateralFilter(L_eq,25,200,50)
img_b = img_g # No bilateral filter

rows = img_b.shape[0]
cols = img_b.shape[1]
thMeaningfulLength = int(2*log(rows*cols)/log(8)+0.5)
gNoise = 1.33333
VMGradient = 70
k=3
gradientMap = np.zeros(img_b.shape)
dx = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=1, dy=0, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
dy = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=0, dy=1, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
if k == 5:
    dx = dx/13.4
    dy = dy/13.4
if k == 7:
    dx = dx/47.5
    dy = dy/47.5
totalNum = 0
histogram = np.zeros(255*8)
for i in range(gradientMap.shape[0]):
    for j in range(gradientMap.shape[1]):
        ptrG = abs(dx[i,j])+abs(dy[i,j])
        if ptrG > gNoise:
            histogram[int(ptrG + 0.5)] += 1
            totalNum +=1
        else:
            ptrG = 0
        gradientMap[i,j] = ptrG
N2 = 0
for i in range(len(histogram)):
    if histogram[i] != 0:
        N2 += histogram[i]*(histogram[i]-1)
pMax = 1/exp((log(N2)/thMeaningfulLength))
pMin = 1/exp((log(N2)/sqrt(cols*rows)))
greaterThan = np.zeros(255*8)
count = 0
for i in range(255*8-1,-1,-1):
    count += histogram[i]
    probabilityGreater = count/totalNum
    greaterThan[i] = probabilityGreater
count = 0
for i in range(255*8-1,-1,-1):
    if greaterThan[i]>=pMax:
        thGradientHigh = i
        break
for i in range(255*8-1,-1,-1):
    if greaterThan[i]>=pMin:
        thGradientLow = i
        break
if thGradientLow <= gNoise:
    thGradientLow = gNoise
thGradientHigh = sqrt(thGradientHigh*VMGradient)
edgemap = cv2.Canny(img_b,thGradientLow,thGradientHigh,3) 
anglePer = np.pi / 8
orientationMap = np.zeros(img_b.shape)
for i in range(orientationMap.shape[0]):
    for j in range(orientationMap.shape[1]):
        ptrO = int((atan2(dx[i,j],-dy[i,j]) + np.pi)/anglePer)
        if ptrO == 16:
            ptrO = 0
        orientationMap[i,j] = ptrO
maskMap = np.zeros(img_b.shape)
gradientPoints = []
gradientValues = []
for i in range(edgemap.shape[0]):
    for j in range(edgemap.shape[1]):
        if edgemap[i,j] == 255:
            maskMap[i,j] = 1
            gradientPoints.append((i,j))
            gradientValues.append(gradientMap[i,j])
gradientPoints = [x for _,x in sorted(zip(gradientValues,gradientPoints))]            
gradientValues.sort()
gradientPoints = gradientPoints[::-1]
gradientValues = gradientValues[::-1] 

# [A] Initial Chains
edgeChainsA = []    
for i in range(len(gradientPoints)):
    x = gradientPoints[i][0]
    y = gradientPoints[i][1]
    if maskMap[x,y] == 0 or maskMap[x,y] == 2:
        continue
    chain = []
    chain.append((x,y))
    while x >= 0 and y >= 0:
        x,y = META.next1(x,y,rows,cols,maskMap,orientationMap)
        if x >= 0 and y >= 0:
            chain.append((x,y))
            maskMap[x,y] = 2
    if len(chain) >= thMeaningfulLength:
        edgeChainsA.append(chain) 
        chain = np.array(chain)       
        
plt.close()
plt.subplot(2,3,1)
plt.imshow(img)
plt.subplot(2,3,2)
plt.imshow(img_eq)
plt.subplot(2,3,3)
plt.imshow(img_g,cmap='gray')     
plt.subplot(2,3,4)
plt.imshow(edgemap)
plt.subplot(2,3,5)
plt.imshow(edgemap)
for chain in edgeChainsA:
    chain = np.array(chain)
    plt.scatter(chain[:,1],chain[:,0],c='r')
plt.subplot(2,3,6)
plt.imshow(img)
for chain in edgeChainsA:
    chain = np.array(chain)
    plt.scatter(chain[:,1],chain[:,0],c='r')