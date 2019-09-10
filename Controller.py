import META
import CANNY
import RECC
import numpy as np
import matplotlib.pyplot as plt

wdir    = r"D:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\ORTHODUMP\Verdonk - Wever Oost"
files   = ["3","1"]
path    = META.initialize(wdir,files)

ps1 = 0.5   #[m]  (0.5)  <First pixelsize>
ps2 = 0.05  #[m]  (0.05) <Second pixelsize>
w   = 25    #[m]  (25)   <Radius template>
md  = 12    #[m]  (12)   <Max displacement>

print("[IMAGE 0]")
gt_0,img_C0,img_b_C0,mask_b_C0,fx_C0,fy_C0,xb_C0,yb_C0,img_F0,fx_F0,fy_F0,xb_F0,yb_F0 = META.correct_ortho(ps1,ps2,path[0])
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(ps1,img_b_C0,mask_b_C0)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsE_C                                     = CANNY.CannyLines(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

img_b_F,mask_b_F,contour_F0                                                           = META.switch_correct_ortho(ps1,ps2,img_F0,edgeChainsE_C)
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(ps2,img_b_F,mask_b_F)
edges0F,edgeChainsA_F,edgeChainsB_F,edgeChainsE_F                                     = CANNY.CannyLines(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

for i in range(1,len(path)):
    print("[IMAGE "+str(i)+"]")
    gt,img_C,img_b_C,mask_b_C,fx_C,fy_C,xb_C,yb_C,img_F,fx_F,fy_F,xb_F,yb_F                 = META.correct_ortho(ps1,ps2,path[i])
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C    = CANNY.CannyPF(ps1,img_b_C,mask_b_C)
    edges1C,edgeChainsA_C,edgeChainsB_C,edgeChainsE_C                                       = CANNY.CannyLines(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)
    x_offset,y_offset,o_xC,o_yC,t_xC,t_yC                                                   = RECC.init_match(ps1,w,md,edges1C,gt,fx_C,fy_C,xb_C,yb_C,edges0C,gt_0,fx_C0,fy_C0,xb_C0,yb_C0,mask_b_C0)

    img_b_F,mask_b_F,contour_F                                                              = META.switch_correct_ortho(ps1,ps2,img_F,edgeChainsE_C)
    edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F    = CANNY.CannyPF(ps2,img_b_F,mask_b_F)
    edges1F,edgeChainsA_F,edgeChainsB_F,edgeChainsE_F                                       = CANNY.CannyLines(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)
    dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,RECC_m,target_l,patch_l,cv = RECC.patch_match(ps1,ps2,w,md,edges1F,gt,fx_F,fy_F,xb_F,yb_F,edges0F,gt_0,fx_F0,fy_F0,xb_F0,yb_F0,contour_F0,x_offset,y_offset)
    gcplist,dist2,origin_x2,origin_y2,target_lon2,target_lat2,o_x2,o_y2,t_x2,t_y2,cvII      = RECC.remove_outliers(ps2,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,cv)
    RECC.georeference(wdir,path[i],files[i],gcplist)

#%% [RECC] Image GCP Comparison (outlier removal)
clist = list(np.random.choice(range(256), size=len(t_x2)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(img_F0)  
plt.scatter(t_y2,t_x2,c=clist)
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(img_F)  
plt.scatter(o_y2,o_x2,c=clist)
    
#%% [RECC] Image GCP Comparison (original)
clist = list(np.random.choice(range(256), size=len(t_x)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(img_F0)  
plt.scatter(t_y,t_x,c=clist)
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(img_F)  
plt.scatter(o_y,o_x,c=clist)

#%% [RECC] Image GCP Comparison (coarse)
clist = list(np.random.choice(range(256), size=len(t_xC)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(edges0C)  
plt.scatter(t_yC,t_xC,c=clist)
#plt.scatter(t_yC+y_offset,t_xC+x_offset,c='r')
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(edges1C)  
plt.scatter(o_yC,o_xC,c=clist)

#%% [RECC] RECC check
fig,ax = plt.subplots()
plt.imshow(RECC_m,cmap='gray')
plt.scatter(t_y,t_x,c='b')
ax.scatter(o_y,o_x,c='r')
for i in range(len(o_y)):
    ax.annotate(str(round(cv[i],2)),(o_y[i]+(7/0.05),o_x[i]-(7/0.05)))

#%% [CANNY] IMAGE OVERLAY CHECK
plt.imshow(img_F)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=1,c='r')
    
#%% [CANNY] EDGEMAP / CHAIN EXTENSION CHECK
plt.imshow(img_Fa)
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='r')
for i in range(len(edgeChainsD_F)):    
    chain = np.array(edgeChainsD_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=2,c='g')
    
#%% [CANNY] STEPWISE CHECK
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
    plt.scatter(chain[:,1],-1*chain[:,0],s=2)