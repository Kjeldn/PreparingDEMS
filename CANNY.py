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

def CannyPF(pixel_size,img_b,mask_b):
    print("Gathering CannyPF edgemap...")
    rows = img_b.shape[0]
    cols = img_b.shape[1]
    thMeaningfulLength = int(2*log(rows*cols)/log(8)+0.5)
    gNoise = 1.33333
    VMGradient = 70
    k=7
    gradientMap = np.zeros(img_b.shape)
    dx = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=1, dy=0, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(src=img_b,ddepth=cv2.CV_16S, dx=0, dy=1, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
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
    edgemap[mask_b>=10**-10]=0   
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
    return edgemap, gradientMap, orientationMap, maskMap, gradientPoints, gradientValues

def CannyLines(pixel_size,edgemap,gradientMap,orientationMap,maskMap,gradientPoints,gradientValues):
    print("Linking edgemap edges...")
    rows = edgemap.shape[0]
    cols = edgemap.shape[1]
    thMeaningfulLength = int(2*log(rows*cols)/log(8)+0.5)
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
    # [C] Alternative Line merging
#    theta_s = thMeaningfulLength / 2 
#    theta_m = 2*tan(2/thMeaningfulLength)
#    metaLinesC = list(copy.deepcopy(metaLinesB))
#    edgeChainsC = list(copy.deepcopy(edgeChainsB))
#    lengthC = list(copy.deepcopy(lengthB))
#    combined_list = [1]
#    i = 0
#    flag = 0
#    while flag <= 1:
#        temp_length = len(metaLinesC)
#        if i >= temp_length:
#            break
#        while len(combined_list) >= 1:
#            x1 = metaLinesC[i][0][0]
#            y1 = metaLinesC[i][1][0]
#            x2 = metaLinesC[i][0][-1]
#            y2 = metaLinesC[i][1][-1]
#            s = metaLinesC[i][2]
#            sd = []
#            d1 = []
#            d2 = []
#            for j in [x for x in range(len(metaLinesC)) if x != i]:
#                s2 = metaLinesB[j][2]
#                sd.append(abs(atan(s)-atan(s2)))
#                a1 = metaLinesC[j][0][0]
#                b1 = metaLinesC[j][1][0]
#                a2 = metaLinesC[j][0][-1]
#                b2 = metaLinesC[j][1][-1]
#                d1.append(sqrt((x1-a2)**2+(y1-b2)**2))
#                d2.append(sqrt((x2-a1)**2+(y2-b1)**2))
#            index = np.where(np.array(sd)<=theta_m)
#            index1 = np.intersect1d(np.where(np.array(d1)<=3/pixel_size),index)
#            index2 = np.intersect1d(np.where(np.array(d2)<=3/pixel_size),index)
#            combined_list = list(index1)
#            combined_list.extend(list(index2))
#            if len(combined_list) >= 1:
#                length_list = []
#                for tempindex in combined_list:
#                    length_list.append(lengthB[tempindex])
#                indexindex = np.where(length_list == max(length_list))[0][0]
#                trueindex = combined_list[indexindex]
#                edgeChainsC[i].extend(edgeChainsC[trueindex])
#                chain = np.array(edgeChainsC[i])
#                m,c = np.polyfit(chain[:,1],chain[:,0],1) 
#                xmin = min(chain[:,1])
#                xmax = max(chain[:,1])
#                xn = np.linspace(xmin,xmax,(max(1,xmax-xmin))*10)
#                yn = np.polyval([m, c], xn)
#                l = sqrt((xn[0]-xn[-1])**2+(yn[0]-yn[-1])**2)
#                metaLinesC[i] = (xn,yn,m,c)
#                lengthC[i] = l
#                del edgeChainsC[trueindex]
#                del metaLinesC[trueindex]
#                del lengthC[trueindex]
#        i += 1
    # [C] Line merging
    theta_s = thMeaningfulLength / 2 
    theta_m = 2*tan(2/thMeaningfulLength)
    connections = []            
    for i in range(len(metaLinesB)):    
        x1 = metaLinesB[i][0][0]
        y1 = metaLinesB[i][1][0]
        x2 = metaLinesB[i][0][-1]
        y2 = metaLinesB[i][1][-1]
        s = metaLinesB[i][2]
        for j in range(i,len(metaLinesB)):
            s2 = metaLinesB[j][2]
            sd = abs(atan(s)-atan(s2))
            if i!= j and sd <= theta_m:
                a1 = metaLinesB[j][0][0]
                b1 = metaLinesB[j][1][0]
                a2 = metaLinesB[j][0][-1]
                b2 = metaLinesB[j][1][-1]
                d1 = sqrt((x1-a2)**2+(y1-b2)**2)
                d2 = sqrt((x2-a1)**2+(y2-b1)**2)
                if d1 <= 3/pixel_size: #and a2 < x1+theta_s: # 2m direct distance
                    p1 = np.array((x1,y1))
                    p2 = np.array((x2,y2))
                    p3 = np.array((a2,b2))
                    d_toline = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                    if d_toline <= 0.2/pixel_size: # 0.1m distance to line
                        connections.append((i,j))
                if d2 <= 3/pixel_size:# and a1 > x2-theta_s: # 2m direct distance
                    p1 = np.array((x1,y1))
                    p2 = np.array((x2,y2))
                    p3 = np.array((a1,b1))
                    d_toline = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                    if d_toline <= 0.2/pixel_size: # 0.1m distance to line
                        connections.append((i,j))           
    chains = []
    connections = np.array(connections)
    connections_c = connections.copy()
    for i in range(len(connections)):
        if all(connections_c[i] == -1):
            continue
        chain = []
        chain.append(connections_c[i][0])
        chain.append(connections_c[i][1])
        connections_c[i]=-1
        while len(set(chain) & set(connections_c[:,0].tolist())) >= 1 or len(set(chain) & set(connections_c[:,1].tolist())) >= 1:
            num1 = set(chain) & set(connections_c[:,0].tolist())
            num2 = set(chain) & set(connections_c[:,1].tolist())
            if len(num1) >= 1:
                index = np.where(connections_c[:,0]==list(num1)[0])[0][0]
            elif len(num2) >= 1:
                index = np.where(connections_c[:,1]==list(num2)[0])[0][0]
            chain.append(connections_c[index][0])
            chain.append(connections_c[index][1])
            connections_c[index]=-1
        chains.append(sorted(list(set(chain))))  
    edgeChainsMerged = []     
    for i in range(len(chains)):
        chain = []
        for j in range(len(chains[i])):
            chain.extend(edgeChainsB[chains[i][j]])
        edgeChainsMerged.append(chain)  
    edgeChainsNonmerged = []
    alledges = np.linspace(0,len(edgeChainsB)-1,len(edgeChainsB)).astype(int)
    merged = np.unique(connections)
    nonmerged = np.array(list(set(alledges)-set(merged)))
    for i in range(len(nonmerged)):
        edgeChainsNonmerged.append(edgeChainsB[nonmerged[i]])
    edgeChainsC = []
    edgeChainsC.extend(edgeChainsMerged)
    edgeChainsC.extend(edgeChainsNonmerged)
    # [C] Line fitting
    metaLinesC = []
    lengthC = []
    for i in range(len(edgeChainsC)):
        chain = np.array(edgeChainsC[i])
        m,c = np.polyfit(chain[:,1],chain[:,0],1) 
        xmin = min(chain[:,1])
        xmax = max(chain[:,1])
        xn = np.linspace(xmin,xmax,(max(1,xmax-xmin))*10)
        yn = np.polyval([m, c], xn)
        l = sqrt((xn[0]-xn[-1])**2+(yn[0]-yn[-1])**2)
        metaLinesC.append((xn,yn,m,c))
        lengthC.append(l)
    lengthC = np.array(lengthC)
    metaLinesC = np.array(metaLinesC)
    edgeChainsC = np.array(edgeChainsC)
    indices = lengthC.argsort()
    indices = indices[::-1]
    metaLinesC = metaLinesC[indices] 
    edgeChainsC = edgeChainsC[indices]
    lengthC = lengthC[indices] 
    # [D] Remove nonsensical length
    if pixel_size >= 0.5:
        metaLinesD = metaLinesC.tolist()
        edgeChainsD = edgeChainsC.tolist()
        lengthD = lengthC.tolist()
        maskMap2 = np.zeros(maskMap.shape)
        for i in range(len(edgeChainsD)):
            for x in edgeChainsD[i]:
                maskMap2[x[0],x[1]]=1 
    else:
        metaLinesD = metaLinesC.tolist()
        edgeChainsD = edgeChainsC.tolist()
        lengthD = lengthC.tolist()
        bullshitlength = 2*thMeaningfulLength
        for i in range(len(metaLinesD)-1,-1,-1):
            if lengthD[i] <= bullshitlength:
                del metaLinesD[i]
                del edgeChainsD[i]
                del lengthD[i]
        maskMap2 = np.zeros(maskMap.shape)
        for i in range(len(edgeChainsD)):
            for x in edgeChainsD[i]:
                maskMap2[x[0],x[1]]=1 
    # [E] Alternative Extending
    edgeChainsE = copy.deepcopy(edgeChainsD)
    if pixel_size <= 0.05:  
        residualMask = (edgemap/255) - maskMap2
        for i in range(len(edgeChainsE)):
            chain = np.array(edgeChainsE[i])
            chain_x = chain[:,0]
            chain_y = chain[:,1]
            indices = chain_y.argsort()
            chain = chain[indices]
            chain_x = chain_x[indices]
            chain_y = chain_y[indices]
            indices = chain_x.argsort()
            chain = chain[indices]
            chain_x = chain_x[indices]
            chain_y = chain_y[indices]
            begin = []
            end = []
            s = metaLinesD[i][2]
            for x in range(len(chain_x)):
                if (chain_x[x]-1 in chain_x)==False and (chain_x[x]-2 in chain_x)==False:
                    begin.append(chain_x[x])
                if (chain_x[x]+1 in chain_x)==False and (chain_x[x]+2 in chain_x)==False:
                    end.append(chain_x[x])
            begin = sorted(list(set(begin)))
            end = sorted(list(set(end)))   
            for m in range(len(begin)):
                if s >= 0:
                    begin_i = min(np.where(chain[:,0]==begin[m])[0])
                else:
                    begin_i = max(np.where(chain[:,0]==begin[m])[0])
                b_x = chain[begin_i,0]
                b_y = chain[begin_i,1]
                for x in range(len(end)-1,-1,-1):
                    if end[x] < b_x:
                        ceil_x  = end[x]
                        break
                    else:
                        ceil_x = 0 
                while b_x >= 0 and b_y >= 0:
                    b_x,b_y = META.next3(b_x,b_y,rows,cols,residualMask,0,s)
                    if b_x <= ceil_x:
                        break
                    if b_x >= 0 and b_y >= 0:
                        edgeChainsE[i].append((b_x,b_y))
                        residualMask[b_x,b_y] = 0
            for n in range(len(end)):
                if s >= 0:
                    end_i = max(np.where(chain[:,0]==end[n])[0])
                else:
                    end_i = min(np.where(chain[:,0]==end[n])[0])
                e_x = chain[end_i,0]
                e_y = chain[end_i,1]
                for x in range(len(begin)):
                    if begin[x] > e_x:
                        ceil_x = end[x]
                        break   
                    else:
                        ceil_x = np.Inf
                while e_x >= 0 and e_y >= 0:
                    e_x,e_y = META.next3(e_x,e_y,rows,cols,residualMask,1,s)
                    if e_x >= ceil_x:
                        break
                    if e_x >= 0 and e_y >= 0:
                        edgeChainsE[i].append((e_x,e_y))
                        residualMask[e_x,e_y] = 0
    edgechainmap = np.zeros(edgemap.shape)
    for chain in edgeChainsE:
        for point in chain:    
            edgechainmap[point[0],point[1]]=1
    return edgechainmap,edgeChainsA,edgeChainsB,edgeChainsC,edgeChainsD,edgeChainsE