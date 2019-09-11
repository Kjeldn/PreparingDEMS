import META
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
import copy
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.filterwarnings("ignore")
from tqdm import tqdm

def CannyPF(pixel_size,img_b,mask_b):
    if pixel_size == 0.05:
        size=6
    else:
        size=5
    pbar1 = tqdm(total=size,position=0,desc="CannyPF   ")
    rows = img_b.shape[0]
    cols = img_b.shape[1]
    thMeaningfulLength = int(2*log(rows*cols)/log(8)+0.5)
    gNoise = 1.33333
    VMGradient = 70
    k=7
    gradientMap = np.zeros(img_b.shape)
    dx = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=1, dy=0, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=0, dy=1, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    pbar1.update(1)
    dx[mask_b>=10**-10]=0
    dy[mask_b>=10**-10]=0
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
    pbar1.update(1)
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
    pbar1.update(1)
    edgemap = cv2.Canny(img_b,thGradientLow,thGradientHigh,3)
    edgemap[mask_b>=10**-10]=0   
    anglePer = np.pi / 8
    orientationMap = np.zeros(img_b.shape)
    for i in range(orientationMap.shape[0]):
        for j in range(orientationMap.shape[1]):
            ptrO = int((atan2(dx[i,j],-dy[i,j]) + np.pi)/anglePer)
            if ptrO == 16:
                ptrO = 0
            orientationMap[i,j] = ptrO
    pbar1.update(1)
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
    pbar1.update(1)
    if pixel_size == 0.05: # SECOND PIXELSIZE
        mask2 = np.zeros(img_b.shape)
        mask2[img_b==0] = 1
        for x in range(edgemap.shape[0]):
            for y in range(edgemap.shape[1]):
                if edgemap[x,y] == 255:
                    if np.sum(mask2[x-2:x+2,y-2:y+2]) >= 1:
                        edgemap[x,y] = 0
        pbar1.update(1)
    pbar1.close()
    return edgemap, gradientMap, orientationMap, maskMap, gradientPoints, gradientValues

def CannyLines(pixel_size,edgemap,gradientMap,orientationMap,maskMap,gradientPoints,gradientValues):
    rows = edgemap.shape[0]
    cols = edgemap.shape[1]
    thMeaningfulLength = int(2*log(rows*cols)/log(8)+0.5)
    pbar2 = tqdm(total=6,position=0,desc="CannyLines")
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
    pbar2.update(1)
    # [B] Splitting orientation shifts
    edgeChainsB = copy.deepcopy(edgeChainsA)
    for i in range(len(edgeChainsB)-1,-1,-1):
        if len(edgeChainsB[i]) >= 2*thMeaningfulLength: 
            orientationchain = []
            for x in edgeChainsB[i]:
                orientationchain.append(orientationMap[x[0],x[1]])
            av = META.moving_average(orientationchain, n=7)
            avchain = np.zeros(len(orientationchain))
            avchain[0:3] = av[0]
            avchain[3:-3] = av
            avchain[-3:] = av[-1]
            d = np.diff(avchain)
            for j in range(len(d)):
                if abs(d[j]) >= 0.3:
                    edgeChainsB.append(edgeChainsB[i][0:j])
                    edgeChainsB.append(edgeChainsB[i][j:])
                    del edgeChainsB[i] 
    edgeChainsB = [x for x in edgeChainsB if x != []] 
    pbar2.update(1)
    # [B] Line fitting     
    metaLinesB = []
    lengthB = []
    for i in range(len(edgeChainsB)):
        chain = np.array(edgeChainsB[i])
        m,c = np.polyfit(chain[:,1],chain[:,0],1) 
        xmin = min(chain[:,1])
        xmax = max(chain[:,1])
        xn = np.linspace(xmin,xmax,(max(1,xmax-xmin))*10)
        yn = np.polyval([m, c], xn)
        l = sqrt((xn[0]-xn[-1])**2+(yn[0]-yn[-1])**2)
        metaLinesB.append((xn,yn,m,c))
        lengthB.append(l)
    lengthB = np.array(lengthB)
    metaLinesB = np.array(metaLinesB)
    edgeChainsB = np.array(edgeChainsB)
    indices = lengthB.argsort()
    indices = indices[::-1]
    metaLinesB = metaLinesB[indices] 
    edgeChainsB = edgeChainsB[indices]
    lengthB = lengthB[indices]   
    pbar2.update(1)
    # [E] Alternative Extending
    edgeChainsE = list(copy.deepcopy(edgeChainsB))
    metaLinesE = list(copy.deepcopy(metaLinesB))
    edgemap_s = (edgemap/255).astype(int)
    residualmap = copy.deepcopy(edgemap_s)
    for i in range(len(edgeChainsE)):
        chain = np.array(edgeChainsE[i])
        chain_x = chain[:,0]
        chain_y = chain[:,1]
        indices = chain_y.argsort()
        chain   = chain[indices]
        chain_x = chain_x[indices]
        chain_y = chain_y[indices]
        indices = chain_x.argsort()
        chain   = chain[indices]
        chain_x = chain_x[indices]
        chain_y = chain_y[indices]
        chain_n = []
        for j in range(len(chain_x)):
            chain_n.append((chain_x[j],chain_y[j]))
            residualmap[chain_x[j],chain_y[j]]=0
        edgeChainsE[i] = chain_n  
    i = -1
    while i <= len(edgeChainsE)-3:
        i += 1
        chain = np.array(edgeChainsE[i])
        s       = metaLinesE[i][2]
        begin   = chain[0,0]
        end     = chain[-1,0]
        # BEGIN
        if s >= 0:
            begin_i = min(np.where(chain[:,0]==begin)[0])
        else:
            begin_i = max(np.where(chain[:,0]==begin)[0])
        b_x = chain[begin_i,0]
        b_y = chain[begin_i,1]
        while b_x >= 0 and b_y >= 0:
            b_x,b_y = META.next4(b_x,b_y,rows,cols,edgemap_s,0,s,edgeChainsE[i])
            if b_x >= 0 and b_y >= 0 and residualmap[b_x,b_y] == 1:            # Extend chain with residual pixel
                edgeChainsE[i].append((b_x,b_y))
                residualmap[b_x,b_y] = 0
            elif b_x >= 0 and b_y >= 0:                                        # Extend chain with another chain
                for j in range(i+1,len(edgeChainsE)):
                    if (b_x,b_y) in edgeChainsE[j]:
                        Tchain   = np.array(edgeChainsE[j])
                        Ts       = metaLinesE[j][2]
                        Tbegin   = Tchain[0,0]
                        Tend     = Tchain[-1,0]
                        erange   = META.rangemaker(Tend,thMeaningfulLength)
                        if b_x in erange:                                      # Appropriate connection 
                            edgeChainsE[i].extend(edgeChainsE[j])
                            if Ts >= 0:
                                begin_i = min(np.where(Tchain[:,0]==Tbegin)[0])
                            else:
                                begin_i = max(np.where(Tchain[:,0]==Tbegin)[0])
                            b_x = Tchain[begin_i,0]
                            b_y = Tchain[begin_i,1]
                            s = Ts
                            del edgeChainsE[j]
                            del metaLinesE[j]
                        else:                                                  # Inappropriate connection
                            b_x = -1
                            b_y = -1
                        break
                    if j == len(edgeChainsE)-1:
                        b_x = -1
                        b_y = -1
        # END
        if s >= 0:
            end_i = max(np.where(chain[:,0]==end)[0])
        else:
            end_i = min(np.where(chain[:,0]==end)[0])
        e_x = chain[end_i,0]
        e_y = chain[end_i,1]
        while e_x >= 0 and e_y >= 0:
            e_x,e_y = META.next4(e_x,e_y,rows,cols,edgemap_s,1,s,edgeChainsE[i])
            if e_x >= 0 and e_y >= 0 and residualmap[e_x,e_y] == 1:
                edgeChainsE[i].append((e_x,e_y))
                residualmap[e_x,e_y] = 0
            elif e_x >= 0 and e_y >= 0:
                for j in range(i+1,len(edgeChainsE)):
                    if (e_x,e_y) in edgeChainsE[j]:
                        Tchain   = np.array(edgeChainsE[j])
                        Ts       = metaLinesE[j][2]
                        Tbegin   = Tchain[0,0]
                        Tend     = Tchain[-1,0]
                        brange   = META.rangemaker(Tbegin,thMeaningfulLength)
                        if e_x in brange:                    
                            edgeChainsE[i].extend(edgeChainsE[j])
                            if Ts >= 0:
                                end_i = max(np.where(Tchain[:,0]==Tend)[0])
                            else:
                                end_i = min(np.where(Tchain[:,0]==Tend)[0])
                            e_x = Tchain[end_i,0]
                            e_y = Tchain[end_i,1]
                            s = Ts
                            del edgeChainsE[j]
                            del metaLinesE[j]
                        else:                                                  
                            e_x = -1
                            e_y = -1
                        break
                    if j == len(edgeChainsE)-1:
                        e_x = -1
                        e_y = -1
    pbar2.update(1)
    # [E] Line fitting     
    metaLinesE = []
    lengthE = []
    for i in range(len(edgeChainsE)):
        chain = np.array(edgeChainsE[i])
        m,c = np.polyfit(chain[:,1],chain[:,0],1) 
        xmin = min(chain[:,1])
        xmax = max(chain[:,1])
        xn = np.linspace(xmin,xmax,(max(1,xmax-xmin))*10)
        yn = np.polyval([m, c], xn)
        l = sqrt((xn[0]-xn[-1])**2+(yn[0]-yn[-1])**2)
        metaLinesE.append((xn,yn,m,c))
        lengthE.append(l)
    lengthE = np.array(lengthE)
    metaLinesE = np.array(metaLinesE)
    edgeChainsE = np.array(edgeChainsE)
    indices = lengthE.argsort()
    indices = indices[::-1]
    metaLinesE = metaLinesE[indices] 
    edgeChainsE = edgeChainsE[indices]
    lengthE = lengthE[indices]   
    pbar2.update(1)
    # [F] Delete
    edgeChainsF = list(copy.deepcopy(edgeChainsE))
    for i in range(len(metaLinesE)-1,-1,-1):
        if lengthE[i] < thMeaningfulLength:
            del edgeChainsF[i]
    # FINALIZE
    edgechainmap = np.zeros(edgemap.shape)
    for chain in edgeChainsF:
        for point in chain:    
            edgechainmap[point[0],point[1]]=1
    pbar2.update(1)
    pbar2.close()
    return edgechainmap,edgeChainsA,edgeChainsB,edgeChainsE