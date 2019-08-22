# Iterative chaining test
import META
import CANNY
import RECC
import numpy as np
import matplotlib.pyplot as plt

wdir = r"E:\ORTHODUMP"
files = ["T0","T1","T2","E0","E1"]

path                                                                                  = META.initialize(wdir,files)
img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,path[3])
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
edgechainmap_C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C  = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

img_F,img_g_F,img_b_F,mask_b_F                                                        = META.switch_correct_ortho(0.5,0.05,path[1],edgeChainsE_C)
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(0.5,img_b_F,mask_b_F)
edgechainmap_F,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F  = CANNY.CannyLines(0.5,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

#%% 1 IMAGE OVERLAY CHECK
plt.imshow(img_F)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=1,c='r')
    
#%% 2 EDGEMAP / CHAIN EXTENSION CHECK
plt.imshow(edgemap_C)
for i in range(len(edgeChainsE_C)):    
    chain = np.array(edgeChainsE_C[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='r')
for i in range(len(edgeChainsD_C)):    
    chain = np.array(edgeChainsD_C[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='g')
    
#%% 3 STEPWISE CHECK
### [A] Initial chains of linked edge pixels
for i in range(len(edgeChainsA_F)):
    chain = np.array(edgeChainsA_F[i])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)
#%% [B] Chains after splitting on orientation
for i in range(len(edgeChainsB_F)):    
    chain = np.array(edgeChainsB_F[i])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)
#%% [C] Chains after merging
for i in range(len(edgeChainsC_F)):    
    chain = np.array(edgeChainsC_F[i])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)  
#%% [D] Chains after removing small lines
for i in range(len(edgeChainsD_F)):    
    chain = np.array(edgeChainsD_F[i])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)   
#%% [E] Chains after extending remaining lines
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)'
    
    #%%
chain = np.array(edgeChainsB_F[0])
plt.scatter(chain[:,1],-1*chain[:,0],s=2)   
chain = np.array(edgeChainsB_F[1])
plt.scatter(chain[:,1],-1*chain[:,0],s=2) 

