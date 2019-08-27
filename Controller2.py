import META
import CANNY
import RECC2
import numpy as np
import matplotlib.pyplot as plt

wdir = r"E:\ORTHODUMP"
files = ["T1","T2"]

path = META.initialize(wdir,files)

file = path[0]
    
img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,file)
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C         = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

img_F,img_g_F,img_b_F,mask_b_0,mask_o_0,fact_x_0,fact_y_0,x_b_0,y_b_0,gt_0            = META.switch_correct_ortho(0.5,0.5,file,edgeChainsE_C)

for i in range(1,2):
    print(i)
    file = path[i]
    
    img_C,img_g_C,img_b_C,mask_b_C                                                         = META.correct_ortho(0.5,file)
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C   = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
    edgechainmap_C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C   = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)
    exec("edges"+str(i)+"C = edgechainmap_C")
    
    img_F,img_g_F,img_b_F,mask_b_F,mask_o,fact_x,fact_y,x_b,y_b,gt                          = META.switch_correct_ortho(0.5,0.5,file,edgeChainsE_C)
    
    dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,RECC_l,target_l,patch_l,cv = RECC2.patch_match(0.5, 50, 10, 10, edges1C, gt, fact_x, fact_y, x_b, y_b, edges0C, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_o_0)
    gcplist,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y                    = RECC2.remove_outliers(50, dist, origin_x, origin_y, target_lon, target_lat, o_x, o_y, t_x, t_y, cv)
    RECC2.georeference(wdir,path[i],files[i],gcplist)
    
#%%
plt.imshow(edges0C,cmap='gray')  
plt.scatter(t_y,t_x,c='r')
plt.scatter(o_y,o_x,c='b')
#%%
plt.imshow(edges0C,cmap='gray')  
ind = np.where(cv<=20)[0]
plt.scatter(t_y[ind],t_x[ind],c='r')
plt.scatter(o_y[ind],o_x[ind],c='b')