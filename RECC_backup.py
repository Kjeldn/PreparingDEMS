import META
import rasterio
import gdal
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from random import randint
from math import cos, sin, asin, sqrt, radians, log, tan, exp, atan2, atan
import warnings
import copy
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#%%
def file_to_edges(i,file_type,path,luma,gamma_correct,pixel_size,extra_blur,ppp,steps,thresholding,sigma):
    if i == 0:
        print("[0%] Getting edges for target image.")
    else:
        print("["+"{:.0f}".format((1+(i-1)*(ppp+3))/steps)+"%] Getting edges for image nr "+str(i)+".")
    if file_type == 0:
        #edges, array, gt, fact_x, fact_y, x_b, y_b, mask = ortho_to_edges(path,luma,gamma_correct,pixel_size,extra_blur,thresholding,sigma)
        #edges, array, gt, fact_x, fact_y, x_b, y_b, mask = ortho_to_edges_new(path,pixel_size,thresholding,sigma,extra_blur)
        edges, array, gt, fact_x, fact_y, x_b, y_b, mask = ortho_to_edges_newnew(path,pixel_size)
    elif file_type == 1:
        edges, gt, fact_x, fact_y, x_b, y_b, mask = dem_to_edges(path,pixel_size)
    return edges, array, gt, fact_x, fact_y, x_b, y_b, mask

def ortho_to_edges(path,luma,gamma_correct,pixel_size,extra_blur,thresholding,sigma):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)   
    if gamma_correct == 1:
        Rlin                           = (R_s**2.2)/255
        Glin                           = (G_s**2.2)/255
        Blin                           = (B_s**2.2)/255
        if   luma == 709:
            Y                          = 0.2126*Rlin + 0.7152*Glin + 0.0722*Blin
        elif luma == 601:
            Y                          = 0.299*Rlin + 0.587*Glin + 0.114*Blin
        elif luma == 240:
            Y                          = 0.212*Rlin + 0.701*Glin + 0.087*Blin
        L                              = 116*Y**(1/3)-16
        arr_sg                         = ((L/np.max(L))*255).astype(np.uint8)
    elif gamma_correct == 0:
        if   luma == 709:
            arr_sg                     = 0.2126*R_s + 0.7152*G_s + 0.0722*B_s
        elif luma == 601:
            arr_sg                     = 0.299*R_s + 0.587*G_s + 0.114*B_s
        elif luma == 240:
            arr_sg                     = 0.212*R_s + 0.701*G_s + 0.087*B_s
    if extra_blur == 1:
            arr_sg                     = cv2.medianBlur(arr_sg.astype(np.uint8),3)
    mask                               = np.zeros(arr_sg.shape)
    mask[arr_sg==255]                  = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)
    if thresholding == 0:
        ht, thresh_im                      = cv2.threshold(arr_sg[arr_sg<=254], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lt = 0.5*ht
    elif thresholding == 1:
        lt = max((1-sigma) * np.median(arr_sg[arr_sg<=254]),0)
        ht = min((1+sigma) * np.median(arr_sg[arr_sg<=254]),255)
    elif thresholding == 2:
        lt = max((1-sigma) * np.mean(arr_sg[arr_sg<=254]),0)
        ht = min((1+sigma) * np.mean(arr_sg[arr_sg<=254]),255)
    edges                              = cv2.Canny(arr_sg,lt,ht)
    edges[mask_b>=10**-10]             = 0 
    fact_x = B.shape[0]/edges.shape[0]
    fact_y = B.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, arr_sg, gt, fact_x, fact_y, x_b, y_b, mask


def ortho_to_edges_new(path,pixel_size,thresholding,sigma,extra_blur):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    img_s                              = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
    mask                               = np.zeros(B_s.shape)
    mask[R_s==255]                     = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)  
    img_s[:,:,0]                       = B_s
    img_s[:,:,1]                       = G_s
    img_s[:,:,2]                       = R_s
    img_s_cielab                       = cv2.cvtColor(img_s, cv2.COLOR_BGR2Lab)
    L                                  = img_s_cielab[:,:,0] 
    hist,bins,trash                    = plt.hist(L[mask_b==0],bins=256)
    cdf                                = hist.cumsum()
    cdf_m                              = np.ma.masked_equal(cdf,0)
    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
    L_eq                               = cdf[L] 
    img_s_cielab_eq                    = img_s_cielab.copy()
    img_s_cielab_eq[:,:,0]             = L_eq   
    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
    img_g                              = cv2.cvtColor(img_s_eq, cv2.COLOR_BGR2GRAY)    
    if thresholding == 0:
        ht, thresh_im                  = cv2.threshold(img_g[mask_b==0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lt                             = 0.5*ht
    elif thresholding == 1:
        lt                             = max((1-sigma) * np.median(img_g[mask_b==0]),0)
        ht                             = min((1+sigma) * np.median(img_g[mask_b==0]),255)
    elif thresholding == 2:
        lt                             = max((1-sigma) * np.mean(img_g[mask_b==0]),0)
        ht                             = min((1+sigma) * np.mean(img_g[mask_b==0]),255)
    if extra_blur == 1:
        img_g                          = cv2.GaussianBlur(img_g,(3,3),0)
    edges                              = cv2.Canny(img_g,lt,ht,L2gradient=True)
    edges[mask_b>=10**-10]             = 0 
    fact_x                             = B.shape[0]/edges.shape[0]
    fact_y                             = B.shape[1]/edges.shape[1]
    x_b                                = edges.shape[0]
    y_b                                = edges.shape[1]
    return edges, img_s_eq, gt, fact_x, fact_y, x_b, y_b, mask

def ortho_to_edges_newnew(path,pixel_size):
    file                               = gdal.Open(path[3])
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    img_s                              = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
    mask                               = np.zeros(B_s.shape)
    mask[R_s==255]                     = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)  
    img_s[:,:,0]                       = B_s
    img_s[:,:,1]                       = G_s
    img_s[:,:,2]                       = R_s
    img_s_cielab                       = cv2.cvtColor(img_s, cv2.COLOR_BGR2Lab)
    L                                  = img_s_cielab[:,:,0] 
    hist,bins,trash                    = plt.hist(L[mask_b==0],bins=256)
    cdf                                = hist.cumsum()
    cdf_m                              = np.ma.masked_equal(cdf,0)
    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
    L_eq                               = cdf[L] 
    img_s_cielab_eq                    = img_s_cielab.copy()
    img_s_cielab_eq[:,:,0]             = L_eq   
    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
    img_g                              = cv2.cvtColor(img_s_eq, cv2.COLOR_BGR2GRAY)
    img_b                              = cv2.bilateralFilter(img_g,21,125,250)
    
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
            x,y = next1(x,y,rows,cols,maskMap,orientationMap)
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
            av = moving_average(orientationchain, n=7)
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
    # # [B] Splitting to meaningful length
    # splitchains = []
    # for i in range(len(edgeChains)-1,-1,-1):
    #     if len(edgeChains[i]) >= 2*thMeaningfulLength:            
    #         chainlist = [edgeChains[i][j: j + 2*thMeaningfulLength] for j in range(0, len(edgeChains[i]), 2*thMeaningfulLength)]
    #         if len(chainlist[-1]) <= thMeaningfulLength:
    #             chainlist[-2].extend(chainlist[-1])
    #             del chainlist[-1]
    #         splitchains.extend(chainlist)
    #         del edgeChains[i]
    # edgeChains.extend(splitchains)              
    # edgeChains = [x for x in edgeChains if x != []]   
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
                if d1 <= 3/pixel_size and a2 < x1: # 2m direct distance
                    p1 = np.array((x1,y1))
                    p2 = np.array((x2,y2))
                    p3 = np.array((a2,b2))
                    d_toline = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                    if d_toline <= 0.08/pixel_size: # 0.1m distance to line
                        connections.append((i,j))
                if d2 <= 3/pixel_size and a1 > x2: # 2m direct distance
                    p1 = np.array((x1,y1))
                    p2 = np.array((x2,y2))
                    p3 = np.array((a1,b1))
                    d_toline = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                    if d_toline <= 0.08/pixel_size: # 0.1m distance to line
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
    # # [E] Store residuals in quad tree
    # residuList = [item for item in gradientPoints if item not in edgeChainsD]
    # spindex = pyqtree.Index(bbox=(0,0,maskMap.shape[0],maskMap.shape[1]))
    #  for item in residuList:
    #     spindex.insert(item, bbox=(item[0],item[1],item[0],item[1]))
    # # [E] Extending chains
    # edgeChainsE = copy.deepcopy(edgeChainsD)
    # d_t = 0.5/pixel_size
    # d_l_t = 0.04/pixel_size
    # for i in range(len(metaLinesD)):
    #     counta = 0
    #     countb = 0
    #     count1 = 0
    #     count2 = 0
    #     last_point = 0
    #     while counta == 0 or countb == 0:
    #         print(i,counta,countb,count1,count2)
    #         if count1 == 0:
    #             b_x = metaLinesD[i][1][0]
    #             b_y = metaLinesD[i][0][0]
    #         b = np.array((b_x,b_y))
    #         if count2 == 0:
    #             e_x = metaLinesD[i][1][-1]
    #             e_y = metaLinesD[i][0][-1]
    #         e = np.array((e_x,e_y))
    #         # Begin:
    #         while counta == 0:
    #             print(i,counta,countb,count1,count2)
    #             overlapbbox = (b_x-d_t,b_y-d_t,b_x+d_t,b_y+d_t)
    #             nearby_points = np.array(spindex.intersect(overlapbbox))
    #             sorting = abs(nearby_points[:,0]-b_x)+abs(nearby_points[:,1]-b_y)
    #             indices = sorting.argsort() 
    #             nearby_points = nearby_points[indices].tolist()
    #             if len(nearby_points) == 0:
    #                 counta += 0.5
    #             for point in nearby_points:
    #                 last_point = nearby_points[-1]
    #                 p = np.array(point)
    #                 d_toline = np.linalg.norm(np.cross(e-b, b-p))/np.linalg.norm(e-b)
    #                 if d_toline <= d_l_t:
    #                     d = sqrt((b_x-point[0])**2+(b_y-point[1])**2)
    #                     if d <= d_t:
    #                         edgeChainsE[i].append(point)
    #                         spindex.remove(tuple(point), bbox=(point[0],point[1],point[0],point[1]))
    #                         count1 += 1
    #                         b_x = point[0]
    #                         b_y = point[1]
    #                         break 
    #             if point == last_point:
    #                 counta += 0.5
    #         # End:
    #         while countb == 0:
    #             print(i,counta,countb,count1,count2)
    #             overlapbbox = (e_x-d_t,e_y-d_t,e_x+d_t,e_y+d_t)
    #             nearby_points = np.array(spindex.intersect(overlapbbox))
    #             sorting = abs(nearby_points[:,0]-e_x)+abs(nearby_points[:,1]-e_y)
    #             indices = sorting.argsort() 
    #             nearby_points = nearby_points[indices].tolist()
    #             if len(nearby_points) == 0:
    #                 countb += 0.5
    #             for point in nearby_points:
    #                 last_point = nearby_points[-1]
    #                 p = np.array(point)
    #                 d_toline = np.linalg.norm(np.cross(e-b, b-p))/np.linalg.norm(e-b)
    #                 if d_toline <= d_l_t:
    #                     d = sqrt((e_x-point[0])**2+(e_y-point[1])**2)
    #                     if d <= d_t:
    #                         edgeChainsE[i].append(point)
    #                         spindex.remove(tuple(point), bbox=(point[0],point[1],point[0],point[1]))
    #                         count2 += 1
    #                         e_x = point[0]
    #                         e_y = point[1]
    #                         break
    #             if point == last_point:
    #                 countb += 0.5                
    # [E] Alternative Extending
    edgeChainsE = copy.deepcopy(edgeChainsD)
    residualMask = (edgemap/255) - maskMap2
    for i in range(len(edgeChainsE)):
        print(i)
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
                b_x,b_y = next2(b_x,b_y,rows,cols,residualMask,0,s)
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
                e_x,e_y = next2(e_x,e_y,rows,cols,residualMask,1,s)
                if e_x >= ceil_x:
                    break
                if e_x >= 0 and e_y >= 0:
                    edgeChainsE[i].append((e_x,e_y))
                    residualMask[e_x,e_y] = 0
                
    #%%
    plt.imshow(img_s_eq)
    #%%
    plt.imshow(gradientMap)
    #%%
    plt.imshow(orientationMap)
    #%%
    plt.imshow(edgemap)
    
    #%% [A] Initial chains of linked edge pixels
    for i in range(len(edgeChainsA)):
        chain = np.array(edgeChainsA[i])
        plt.scatter(chain[:,1],-1*chain[:,0],s=2)
    #%% [B] Chains after splitting on orientation
    for i in range(len(edgeChainsB)):    
        chain = np.array(edgeChainsB[i])
        plt.scatter(chain[:,1],-1*chain[:,0],s=2)
    #%% [B]
    for i in range(len(metaLinesB)):
        xn = metaLinesB[i][0]
        yn = -1*metaLinesB[i][1]
        plt.plot(xn,yn)
    #%% [C] Chains after merging
    for i in range(len(edgeChainsC)):    
        chain = np.array(edgeChainsC[i])
        plt.scatter(chain[:,1],-1*chain[:,0],s=2)
    #%% [C]
    for i in range(len(metaLinesC)):
        xn = metaLinesC[i][0]
        yn = -1*metaLinesC[i][1]
        plt.plot(xn,yn)
    #%% [D] Chains after removing small lines
    for i in range(len(edgeChainsD)):    
        chain = np.array(edgeChainsD[i])
        plt.scatter(chain[:,1],-1*chain[:,0],s=2)
    #%% [D]
    plt.imshow(edgemap)
    for i in range(len(metaLinesD)):
        xn = metaLinesD[i][0]
        yn = metaLinesD[i][1]
        plt.plot(xn,yn)    
    #%% [E] Chains after extending remaining lines
    for i in range(len(edgeChainsE)):    
        chain = np.array(edgeChainsE[i])
        plt.scatter(chain[:,1],-1*chain[:,0],s=2)

    #%% Overlay Image
    plt.imshow(edgemap)
    for i in range(len(edgeChainsE)):    
        chain = np.array(edgeChainsE[i])
        plt.scatter(chain[:,1],chain[:,0],s=1,c='r')
        
    #%%
    plt.imshow(img_s)
    for i in range(len(edgeChainsE)):    
        chain = np.array(edgeChainsE[i])
        plt.scatter(chain[:,1],chain[:,0],s=2,c='r')
    for i in range(len(edgeChainsD)):    
        chain = np.array(edgeChainsD[i])
        plt.scatter(chain[:,1],chain[:,0],s=2,c='g')
    #chain = np.array(edgeChainsD[-20])
    #plt.scatter(chain[:,1],chain[:,0],s=2,c='b')
        
    #%%
    chain = np.array(edgeChainsC[1])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)

    #%%
    fact_x                             = B.shape[0]/edgemap.shape[0]
    fact_y                             = B.shape[1]/edgemap.shape[1]
    x_b                                = edgemap.shape[0]
    y_b                                = edgemap.shape[1]
    return edgemap, img_s_eq, gt, fact_x, fact_y, x_b, y_b, mask


def ortho_to_color(path,pixel_size):
    file                               = gdal.Open(path)
    gt                                 = file.GetGeoTransform()
    R                                  = file.GetRasterBand(1).ReadAsArray()
    G                                  = file.GetRasterBand(2).ReadAsArray()
    B                                  = file.GetRasterBand(3).ReadAsArray()
    x_s, y_s                           = calc_pixsize(R,gt)
    R_s                                = cv2.resize(R,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    G_s                                = cv2.resize(G,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    B_s                                = cv2.resize(B,(int(B.shape[1]*(y_s/pixel_size)), int(B.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    img_s                              = np.zeros([B_s.shape[0],B_s.shape[1],3], np.uint8)
    mask                               = np.zeros(B_s.shape)
    mask[R_s==255]                     = 1
    mask_b                             = cv2.GaussianBlur(mask,(5,5),0)  
    img_s[:,:,0]                       = B_s
    img_s[:,:,1]                       = G_s
    img_s[:,:,2]                       = R_s
    img_s_cielab                       = cv2.cvtColor(img_s, cv2.COLOR_BGR2Lab)
    L                                  = img_s_cielab[:,:,0] 
    hist,bins,trash                    = plt.hist(L[mask_b==0],bins=256)
    cdf                                = hist.cumsum()
    cdf_m                              = np.ma.masked_equal(cdf,0)
    cdf_m                              = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
    cdf                                = np.ma.filled(cdf_m,0).astype(np.uint8)     
    L_eq                               = cdf[L] 
    img_s_cielab_eq                    = img_s_cielab.copy()
    img_s_cielab_eq[:,:,0]             = L_eq   
    img_s_eq                           = cv2.cvtColor(img_s_cielab_eq, cv2.COLOR_Lab2BGR)
    edgemap = np.zeros(img_s_eq.shape)    
    fact_x                             = B.shape[0]/edgemap.shape[0]
    fact_y                             = B.shape[1]/edgemap.shape[1]
    x_b                                = edgemap.shape[0]
    y_b                                = edgemap.shape[1]
    return edgemap, img_s_eq, gt, fact_x, fact_y, x_b, y_b, mask
    
def dem_to_edges(path,pixel_size):
    s                                  = 5
    file                               = gdal.Open(path)
    band                               = file.GetRasterBand(1)
    gt                                 = file.GetGeoTransform()
    arr                                = band.ReadAsArray()
    x_s, y_s                           = calc_pixsize(arr,gt)
    arr_s                              = cv2.resize(arr,(int(arr.shape[1]*(y_s/pixel_size)), int(arr.shape[0]*(x_s/pixel_size))),interpolation = cv2.INTER_AREA)
    mask                               = np.zeros(arr_s.shape)
    mask[arr_s<=0.8*np.nanmin(arr_s)]  = 1
    mask_b                             = cv2.GaussianBlur(mask,(s,s),0)
    mask_hb                            = cv2.GaussianBlur(mask,(21,21),0) 
    arr_sc                             = arr_s.copy()
    arr_sc[mask_b>=10**-10]            = np.NaN
    while np.nanmax(arr_sc)-np.nanmin(arr_sc) >= 100:
        s = s+2
        mask_b                         = cv2.GaussianBlur(mask,(s,s),0)
        arr_sc[mask_b>=10**-10]        = np.NaN
    mask_hb                            = cv2.GaussianBlur(mask,(s+6,s+6),0) 
    sort                               = np.unique(arr_sc[~np.isnan(arr_sc)])
    sortofmedian                       = sort[int(len(sort)/2)]
    std                                = np.std(arr_s[~np.isnan(arr_sc)])
    cap                                = sortofmedian + 1.5*std
    arr_scc                            = arr_sc.copy()
    arr_scc[arr_scc>=cap]              = cap
    arr_sccg                           = arr_scc.copy()
    un                                 = np.nanmin(arr_sccg)
    arr_sccg                           = arr_sccg-un
    up                                 = np.nanmax(arr_sccg)
    arr_sccg                           = (arr_sccg/up)*255
    arr_sccgb                          = cv2.GaussianBlur(arr_sccg,(3,3),0)
    arr_sccgbf                         = arr_sccgb.copy()
    arr_sccgbf[np.isnan(arr_sccgbf)]   = np.nanmean(arr_sccgbf)
    arr_sccgbf                         = arr_sccgbf.astype(np.uint8)    
    arr_sccgbfa                        = cv2.adaptiveThreshold(arr_sccgbf,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
    arr_sccgbfa[mask_hb>=10**-10]      = 0
    edges                              = cv2.medianBlur(arr_sccgbfa,5) 
    fact_x = arr.shape[0]/edges.shape[0]
    fact_y = arr.shape[1]/edges.shape[1]
    x_b    = edges.shape[0]
    y_b    = edges.shape[1]
    return edges, arr_sc, gt, fact_x, fact_y, x_b, y_b, mask
                
def patch_match(i, edges, gt, fact_x, fact_y, x_b, y_b, mask, edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, s, it_cancel, it_max):
    print("["+"{:.0f}".format((2+(i-1)*(ppp+3))/steps)+"%] Matching patches for image nr "+str(i)+".")
    sumedges_0 = np.zeros(edges_0.shape)
    for x in range(w,x_b_0-w):
        for y in range(w,y_b_0-w):
            sumedges_0[x,y] = sum(sum(edges_0[x-w:x+w,y-w:y+w]))
    minfeatures = np.max(sumedges_0)*0.1
    mask_0_c   = mask_0.copy()
    RECC_s     = np.zeros(edges.shape)
    dist       = np.zeros(ppp)
    dist_lon   = np.zeros(ppp)
    dist_lat   = np.zeros(ppp)
    origin_x   = np.zeros(ppp)
    origin_y   = np.zeros(ppp)
    target_lon = np.zeros(ppp)
    target_lat = np.zeros(ppp)
    o_x        = np.zeros(ppp)
    o_y        = np.zeros(ppp)
    t_x        = np.zeros(ppp)
    t_y        = np.zeros(ppp)
    j=-1
    it = 0
    if it_cancel == 1:
        while j <= ppp-2 and it*it_cancel <= it_max*ppp:
            x_i_0 = randint(w,x_b_0-w)
            y_i_0 = randint(w,y_b_0-w)
            check1 = mask_0_c[x_i_0,y_i_0]
            check2 = sumedges_0[x_i_0,y_i_0]
            if check1 <= 0 and check2 >= minfeatures:
                it = it+1
                target = edges_0[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
                sum_target = np.sum(target)
                RECC_s.fill(np.NaN)
                for x in range(max(w,x_i_0-s*w),min(x_b-w,x_i_0+s*w)):
                    for y in range(max(w,y_i_0-s*w),min(y_b-w,y_i_0+s*w)):
                        patch = edges[x-w:x+w,y-w:y+w]
                        RECC_s[x,y]=np.sum(np.multiply(target,patch))/(sum_target+np.sum(patch))           
                max_one  = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-1)[-1]
                max_n    = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-4-1)[-4-1]
                x_i    = np.where(RECC_s >= max_one)[1][0]
                y_i    = np.where(RECC_s >= max_one)[0][0]
                x_n      = np.where(RECC_s >= max_n)[1][0:-1]
                y_n      = np.where(RECC_s >= max_n)[0][0:-1]
                cv_score = sum(np.sqrt(np.square(x_i-x_n)+np.square(y_i-y_n)))          
                if cv_score <= cv_max:
                    lon = gt[0] + gt[1]*x_i*fact_x + gt[2]*y_i*fact_y
                    lat = gt[3] + gt[4]*x_i*fact_x + gt[5]*y_i*fact_y
                    lon_0 = gt_0[0] + gt_0[1]*y_i_0*fact_x_0 + gt_0[2]*x_i_0*fact_y_0
                    lat_0 = gt_0[3] + gt_0[4]*y_i_0*fact_x_0 + gt_0[5]*x_i_0*fact_y_0
                    dst = calc_distance(lat,lon,lat_0,lon_0)
                    if dst <= dst_max:
                        j=j+1
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Succesful patch-match nr "+str(j+1)+" of "+str(ppp)+".")
                        mask_0_c[x_i_0-v:x_i_0+v,y_i_0-v:y_i_0+v]=1
                        dist[j]       = dst
                        dist_lon[j]   = lon_0-lon
                        dist_lat[j]   = lat_0-lat
                        origin_x[j]   = x_i*fact_x
                        origin_y[j]   = y_i*fact_y
                        target_lon[j] = lon_0
                        target_lat[j] = lat_0
                        o_x[j] = x_i
                        o_y[j] = y_i
                        t_x[j] = x_i_0
                        t_y[j] = y_i_0
                    else:
                        print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+","+"{:.1f}".format(dst)+") Match failed.")
                else:
                    print("["+"{:.0f}".format(((3+j)+(i-1)*(ppp+3))/steps)+"%] ("+"{:.0f}".format(cv_score)+",-) Match failed.")      
        dist = dist[dist!=0]
        dist_lon = dist_lon[dist_lon!=0]
        dist_lat = dist_lat[dist_lat!=0]
        origin_x = origin_x[origin_x!=0]
        origin_y = origin_y[origin_y!=0]
        target_lon = target_lon[target_lon!=0]
        target_lat = target_lat[target_lat!=0]
        o_x = o_x[o_x!=0]
        o_y = o_y[o_y!=0]
        t_x = t_x[t_x!=0]
        t_y = t_y[t_y!=0] 
        # flip the target x and y for some reason:
        t_x_temp = t_y
        t_y = t_x
        t_x = t_x_temp
    return dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y

def remove_outliers(i, ppp, conf, steps, outlier_type, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y):
    gcplist = " "
    size = len(dist)
    if outlier_type == 0:
        flier_indices = np.zeros(1)
        while len(flier_indices) >= 1:
            box_dst = plt.boxplot(dist)
            box_lon = plt.boxplot(dist_lon)
            box_lat = plt.boxplot(dist_lat)
            fliers_dst = box_dst["fliers"][0].get_data()[1]
            fliers_lon = box_lon["fliers"][0].get_data()[1]
            fliers_lat = box_lat["fliers"][0].get_data()[1]
            flier_indices_dst = np.zeros(len(fliers_dst))
            flier_indices_lon = np.zeros(len(fliers_lon))
            flier_indices_lat = np.zeros(len(fliers_lat))
            for j in range(len(fliers_dst)):
                flier_indices_dst[j] = np.where(dist==fliers_dst[j])[0][0]
            for j in range(len(fliers_lon)):
                flier_indices_lon[j] = np.where(dist_lon==fliers_lon[j])[0][0]
            for j in range(len(fliers_lat)):
                flier_indices_lat[j] = np.where(dist_lat==fliers_lat[j])[0][0]
            flier_indices = np.union1d(flier_indices_lon,flier_indices_lat)
            flier_indices = (np.union1d(flier_indices,flier_indices_dst)).astype(int)  
            dist       = np.delete(dist,flier_indices)
            dist_lon   = np.delete(dist_lon,flier_indices)
            dist_lat   = np.delete(dist_lat,flier_indices)
            origin_x   = np.delete(origin_x,flier_indices)
            origin_y   = np.delete(origin_y,flier_indices)
            target_lon = np.delete(target_lon,flier_indices)
            target_lat = np.delete(target_lat,flier_indices) 
            o_x        = np.delete(o_x,flier_indices)
            o_y        = np.delete(o_y,flier_indices)
            t_x        = np.delete(t_x,flier_indices)
            t_y        = np.delete(t_y,flier_indices)
    elif outlier_type == 1:
        if   conf == 95:
            s = 5.991
        elif conf == 90:
            s = 4.605
        elif conf == 80:
            s = 3.219
        elif conf == 75:
            s = 2.770
        elif conf == 50:
            s = 1.388
        d_x = t_x - o_x
        d_y = t_y - o_y
        d_x_m = d_x - np.median(d_x)
        d_y_m = d_y - np.median(d_y)        
        indices = ((d_x_m/sqrt(np.var(d_x_m)))**2 + (d_y_m/sqrt(np.var(d_y_m)))**2 >= s)    
        dist       = dist[~indices]
        dist_lon   = dist_lon[~indices]
        dist_lat   = dist_lat[~indices]
        origin_x   = origin_x[~indices]
        origin_y   = origin_y[~indices]
        target_lon = target_lon[~indices]
        target_lat = target_lat[~indices]
        o_x        = o_x[~indices]
        o_y        = o_y[~indices]
        t_x        = t_x[~indices]
        t_y        = t_y[~indices]
    print("["+"{:.0f}".format(((2+ppp)+(i-1)*(ppp+3))/steps)+"%] Removed "+str(size-len(dist))+" outliers.")      
    for k in range(len(origin_x)):
        gcplist = gcplist+"-gcp "+str(origin_x[k])+" "+str(origin_y[k])+" "+str(target_lon[k])+" "+str(target_lat[k])+" "        
    return gcplist, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y
      
def georeference(i,wdir,ppp,path,file,steps,gcplist):
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Georeferencing image nr "+str(i)+".")
    path1 = wdir+"\\temp.tif"
    path2 = wdir+"\\"+file+"_adjusted.tif"
    if os.path.isfile(path1.replace("\\","/")):
        os.remove(path1)
    if os.path.isfile(path2.replace("\\","/")):
        os.remove(path2)
    os.system("gdal_translate -a_srs EPSG:4326 -of GTiff"+gcplist+"\""+path+"\" \""+path1+"\"")
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Succesful translate, warping...")
    os.system("gdalwarp -r cubicspline -tps -co COMPRESS=NONE \""+path1+"\" \""+path2+"\"")    
    
