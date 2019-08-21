# Iterative chaining test
import META
import CANNY
import RECC
import numpy as np
import matplotlib.pyplot as plt

wdir = r"E:\ORTHODUMP"
files = ["T0","T1","T2","E0","E1"]

path                                                                                  = META.initialize(wdir,files)
img_s_C,img_b_C,mask_b_C                                                              = META.correct_ortho(0.5,path[0])
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
edgechainmap_C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C  = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C,edgemap_C)
img_s_F,img_b_F,mask_b_F                                                              = META.correct_ortho(0.05,path[0])
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(0.5,img_b_F,mask_b_F)
edgemap_FA,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F = META.pixelswitch(0.5,0.05,edgeChainsE_C,edgemap_F, gradientMap_F, orientationMap_F, maskMap_F, gradientPoints_F, gradientValues_F)
edgechainmap_F,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F  = CANNY.CannyLines(0.5,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F,edgemap_FA)

#%% 1 IMAGE OVERLAY CHECK
plt.imshow(img_s_F)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=1,c='r')
    
#%% 2 EDGEMAP / CHAIN EXTENSION CHECK
plt.imshow(edgemap_FA)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='r')
for i in range(len(edgeChainsD_F)):    
    chain = np.array(edgeChainsD_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='g')
    
#%% 3 STEPWISE CHECK
### [A] Initial chains of linked edge pixels
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