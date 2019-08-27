import META
import CANNY
import RECC2
import numpy as np
import matplotlib.pyplot as plt

wdir = r"E:\ORTHODUMP"
files = ["T1","T2"]

path = META.initialize(wdir,files)

temp_path = path[0]
    
img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,temp_path)
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C         = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)
    
img_F,img_g_F,img_b_F,mask_b_0,mask_o_0,fact_x_0,fact_y_0,x_b_0,y_b_0,gt_0            = META.switch_correct_ortho(0.5,0.05,temp_path,edgeChainsE_C)
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(0.05,img_b_F,mask_b_0)
edges0F,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F         = CANNY.CannyLines(0.05,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

for i in range(1,2):
    print(i)
    temp_path = path[i]
    
    img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,temp_path)
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
    edgechainmap_C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C  = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)
    exec("edges"+str(i)+"C = edgechainmap_C")
    
    img_F,img_g_F,img_b_F,mask_b_F,mask_o,fact_x,fact_y,x_b,y_b,gt                        = META.switch_correct_ortho(0.5,0.05,temp_path,edgeChainsE_C)
    edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(0.05,img_b_F,mask_b_F)
    edges,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F           = CANNY.CannyLines(0.05,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)
    exec("edges"+str(i)+"F = edges")

    dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,RECC_l,target_l,patch_l,cv = RECC2.patch_match(0.05, 500, 500, 10, edges1F, gt, fact_x, fact_y, x_b, y_b, edges0F, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_o_0)
    gcplist,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y                    = RECC2.remove_outliers(50, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv)
    RECC2.georeference(wdir,path[i],files[i],gcplist)




#%%
plt.imshow(edges0F,cmap='gray')  
plt.scatter(t_y,t_x,c='r')
plt.scatter(o_y,o_x,c='b')
#%%
plt.imshow(edges0F,cmap='gray')  
ind = np.where(cv<=20)[0]
plt.scatter(t_y[ind],t_x[ind],c='r')
plt.scatter(o_y[ind],o_x[ind],c='b')
    
#%% 1 IMAGE OVERLAY CHECK
plt.imshow(edges_0,cmap='gray')
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=1,c='y')
plt.scatter(t_y[index],t_x[index],c='r')
plt.scatter(o_y[index],o_x[index],c='b')
    
#%% 2 EDGEMAP / CHAIN EXTENSION CHECK
plt.imshow(img_F)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='r')
for i in range(len(edgeChainsD_F)):    
    chain = np.array(edgeChainsD_F[i])
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
    