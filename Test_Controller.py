import META
import CANNY
import RECC
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

wdir    = r"\\STAMPERTJE\Data\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\ORTHODUMP\Extra"
files   = ["E0","E1"]
path = META.initialize(wdir,files)

ps1 = 0.5   #[m]   (0.5)
ps2 = 0.05  #[m]   (0.05)
w   = 500   #[pix] (500)
md  = 12    #[m]   (12)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
logname = files[0]+"_LOG.txt"
f = open(logname, 'w')
original1 = sys.stdout
original2 = sys.stderr    
sys.stdout = Tee(sys.stdout, f)
sys.stderr = Tee(sys.stderr, f)

print("[IMAGE 0]")
img_C0,img_g_C,img_b_C,mask_b_C,gt_0,img_Fa0,fact_x_0,fact_y_0,x_b_0,y_b_0            = META.correct_ortho(ps1,ps2,path[0])
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(ps1,img_b_C,mask_b_C)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsE_C                                     = CANNY.CannyLines2(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

plt.imshow(img_C0)
temp = edges0C
temp[temp==0]=np.NaN
temp[0,0]=0
plt.imshow(temp,cmap='Wistia')
plt.savefig(files[0]+"_0C.png",dpi = 500)
plt.clf()

img_F,img_g_F,img_b_F,mask_b_0,mask_o_0                                               = META.switch_correct_ortho(ps1,ps2,img_Fa0,edgeChainsE_C)
edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F  = CANNY.CannyPF(ps2,img_b_F,mask_b_0)
edges0F,edgeChainsA_F,edgeChainsB_F,edgeChainsE_F                                     = CANNY.CannyLines2(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

plt.imshow(img_Fa0)
temp = cv2.GaussianBlur(edges0F,(5,5),1)
temp = edges0F
temp[temp==0]=np.NaN
temp[0,0]=0
plt.imshow(temp,cmap='Wistia')
plt.savefig(files[0]+"_0F.png",dpi = 1000)
plt.clf()

for i in range(1,len(path)):
    print("[IMAGE "+str(i)+"]")
    img_C,img_g_C,img_b_C,mask_b_C,gt,img_Fa,fact_x,fact_y,x_b,y_b                          = META.correct_ortho(ps1,ps2,path[i])
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C    = CANNY.CannyPF(ps1,img_b_C,mask_b_C)
    edges1C,edgeChainsA_C,edgeChainsB_C,edgeChainsE_C                                       = CANNY.CannyLines2(ps1,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

    
    plt.imshow(img_C)
    temp = edges1C
    temp[temp==0]=np.NaN
    temp[0,0]=0
    plt.imshow(temp,cmap='Wistia')
    plt.savefig(files[0]+"_"+str(i)+"C.png",dpi = 500)
    plt.clf()

    img_F,img_g_F,img_b_F,mask_b_F,mask_o                                                   = META.switch_correct_ortho(ps1,ps2,img_Fa,edgeChainsE_C)
    edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F    = CANNY.CannyPF(ps2,img_b_F,mask_b_F)
    edges1F,edgeChainsA_F,edgeChainsB_F,edgeChainsE_F                                       = CANNY.CannyLines2(ps2,edgemap_F,gradientMap_F,orientationMap_F,maskMap_F,gradientPoints_F,gradientValues_F)

    plt.imshow(img_Fa)
    temp = cv2.GaussianBlur(edges1F,(5,5),1)
    temp[temp==0]=np.NaN
    temp[0,0]=0
    plt.imshow(temp,cmap='Wistia')
    plt.savefig(files[0]+"_"+str(i)+"F.png",dpi = 1000)
    plt.clf()
    
    dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,RECC_m,target_l,patch_l,cv = RECC.patch_match(ps2,w,md,edges1F,gt,fact_x,fact_y,x_b,y_b,edges0F,gt_0,fact_x_0,fact_y_0,x_b_0,y_b_0,mask_o_0)
    gcplist,dist2,origin_x2,origin_y2,target_lon2,target_lat2,o_x2,o_y2,t_x2,t_y2,cvII      = RECC.remove_outliers3(ps2,dist,origin_x,origin_y,target_lon,target_lat,o_x,o_y,t_x,t_y,cv)
    RECC.georeference(wdir,path[i],files[i],gcplist)
    
    clist = list(np.random.choice(range(256), size=len(t_y)))
    plt.subplot(1,2,1)
    plt.title('Orthomosaic 1')
    plt.imshow(img_Fa0)  
    plt.scatter(t_y,t_x,c=clist)
    plt.subplot(1,2,2)
    plt.title('Orthomosaic 2')
    plt.imshow(img_Fa)  
    plt.scatter(o_y,o_x,c=clist)
    plt.savefig(files[0]+"_"+str(i)+"_RECC1.png",dpi = 1000)
    plt.clf()    
    
    clist = list(np.random.choice(range(256), size=len(t_y2)))
    plt.subplot(1,2,1)
    plt.title('Orthomosaic 1')
    plt.imshow(img_Fa0)  
    plt.scatter(t_y2,t_x2,c=clist)
    plt.subplot(1,2,2)
    plt.title('Orthomosaic 2')
    plt.imshow(img_Fa)  
    plt.scatter(o_y2,o_x2,c=clist)
    plt.savefig(files[0]+"_"+str(i)+"_RECC2.png",dpi = 1000)
    plt.clf()
    
sys.stdout = original1
sys.stderr = original2
f.close()

#%% [RECC] Image GCP Comparison
clist = list(np.random.choice(range(256), size=len(t_x2)))
plt.subplot(1,2,1)
plt.title('Orthomosaic 1')
plt.imshow(img_Fa0)  
plt.scatter(t_y2,t_x2,c=clist)
plt.subplot(1,2,2)
plt.title('Orthomosaic 2')
plt.imshow(img_Fa)  
plt.scatter(o_y2,o_x2,c=clist)
    
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