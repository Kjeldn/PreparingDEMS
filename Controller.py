import META
import CANNY
import RECC
import numpy as np
import matplotlib.pyplot as plt

wdir  = r"E:\ORTHODUMP"
files = ["T0","T1"]
path  = META.initialize(wdir,files)

ps1 = 0.5  #[m]   (0.5)
ps2 = 0.05 #[m]   (0.05)
w   = 500  #[pix] (500)
s   = 100  #[pix] (1000)
md  = 10   #[m]   (7)
cvt = 4    #[pix] (1.5)
ci  = 50   #[%]   (50/75/80/90/95)

print("[IMAGE 0]")
img_C0,img_g_C,img_b_C,mask_b_C,gt_0,img_Fa0,fact_x_0,fact_y_0,x_b_0,y_b_0             = META.correct_ortho(ps1,ps2,path[0])
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(ps1,img_b_C,mask_b_C)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C         = CANNY.CannyLines(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

img_F,img_g_F,img_b_F,mask_b_0,mask_o_0                                               = META.switch_correct_ortho(ps1,ps2,img_Fa0,edgeChainsE_C)
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(ps2,img_b_F,mask_b_0)
edges0F,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F         = CANNY.CannyLines(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

for i in range(1,len(path)):
    print("[IMAGE "+str(i)+"]")
    img_C,img_g_C,img_b_C,mask_b_C,gt,img_Fa,fact_x,fact_y,x_b,y_b                          = META.correct_ortho(ps1,ps2,path[i])
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C    = CANNY.CannyPF(ps1,img_b_C,mask_b_C)
    edges1C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C           = CANNY.CannyLines(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

    img_F,img_g_F,img_b_F,mask_b_F,mask_o                                                   = META.switch_correct_ortho(ps1,ps2,img_Fa,edgeChainsE_C)
    edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F    = CANNY.CannyPF(ps2,img_b_F,mask_b_F)
    edges1F,edgeChainsA_F,edgeChainsB_F,edgeChainsC_F,edgeChainsD_F,edgeChainsE_F           = CANNY.CannyLines(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

    dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,RECC_m,target_l,patch_l,cv = RECC.patch_match(ps2,w,s,md,edges1F,gt,fact_x,fact_y,x_b,y_b,edges0F,gt_0,fact_x_0,fact_y_0,x_b_0,y_b_0,mask_o_0)
    gcplist,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,cv                 = RECC.remove_outliers(ci,cvt,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,cv)
    RECC.georeference(wdir,path[i],files[i],gcplist)

#%% [RECC] Image GCP Comparison
clist = list(np.random.choice(range(256), size=len(dist)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(img_Fa0)  
plt.scatter(t_y,t_x,c=clist)
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(img_Fa)  
plt.scatter(o_y,o_x,c=clist)
    
#%% [RECC] Edgemap GCP Comparison
clist = list(np.random.choice(range(256), size=len(dist)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(edges0F)  
plt.scatter(t_y,t_x,c=clist)
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(edges1F)  
plt.scatter(o_y,o_x,c=clist)

#%% [RECC] RECC check
fig,ax = plt.subplots()
plt.imshow(RECC_m,cmap='gray')
plt.scatter(t_y,t_x,c='b')
ax.scatter(o_y,o_x,c='r')
for i in range(len(o_y)):
    ax.annotate(str(round(cv[i],2)),(o_y[i]+(7/0.05),o_x[i]-(7/0.05)))

#%% [CANNY] IMAGE OVERLAY CHECK
plt.imshow(edges_0,cmap='gray')
for i in range(len(edgeChainsE_F)):    
    chain = np.array(edgeChainsE_F[i])
    plt.scatter(chain[:,1],chain[:,0],s=1,c='y')
plt.scatter(t_y[index],t_x[index],c='r')
plt.scatter(o_y[index],o_x[index],c='b')
    
#%% [CANNY] EDGEMAP / CHAIN EXTENSION CHECK
plt.imshow(img_F)
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
    