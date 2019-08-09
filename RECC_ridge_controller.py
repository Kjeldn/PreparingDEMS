#RIDGES
wdir = r"Z:\VanBovenDrive\VanBoven MT\500 Projects\Student Assignments\Interns\190806_DEMs_Boomgaard\ridges"
files = ["0","1"]

file_type      = 0         # [0: Orthomosaic] [1: DEM]
outlier_type   = 0         # [0: Boxplot] [1: 99%]
gamma_correct  = 1         # {0,1}
extra_blur     = 1         # {0,1}
luma           = 709       # {240,601,709}
pixel_size     = 0.6       # One pixel will be <...> m by <...> m.
ppp            = 10        # Use <...> points per photo to georeference.
cv_max         = 40        # Approve matches only if CV score is under <...> .
dst_max        = 20        # Approve matches only if distance is under <...> .
w              = 30        # Patch-match will be 2*<...> pixels by 2*<...> pixels.
v              = 30        # Masking area will be 2*<...> pixels by 2*<...> pixels.
s              = 4         # RECC search range distance is capped at {<...>*w*pixel_size} [m].
it_cancel      = 1         # {0,1} 
it_max         = 2         # Max number of iterations per match

import RECC
import numpy as np
import matplotlib.pyplot as plt
import gdal
import cv2

#%%
path,steps = RECC.initialize(wdir,files,ppp)
for i in range(len(path)):
    file                               = gdal.Open(path[i])
    band                               = file.GetRasterBand(1)
    exec("gt_"+str(i)+"                = file.GetGeoTransform()")
    arr                                = band.ReadAsArray()
    exec("x_s, y_s                     = RECC.calc_pixsize(arr,gt_"+str(i)+")")
    exec("edges_"+str(i)+"             = cv2.resize(arr,(int(arr.shape[1]*(y_s/0.6)), int(arr.shape[0]*(x_s/0.6))),interpolation = cv2.INTER_NEAREST)")
    exec("fact_x_"+str(i)+" = arr.shape[0]/edges_"+str(i)+".shape[0]")
    exec("fact_y_"+str(i)+" = arr.shape[1]/edges_"+str(i)+".shape[1]")
    exec("x_b_"+str(i)+"    = edges_"+str(i)+".shape[0]")
    exec("y_b_"+str(i)+"    = edges_"+str(i)+".shape[1]")
    exec("mask_"+str(i)+"   = np.zeros(edges_"+str(i)+".shape)")
    if i >= 1:
        exec("dist_"+str(i)+", dist_lon_"+str(i)+", dist_lat_"+str(i)+", origin_x_"+str(i)+", origin_y_"+str(i)+", target_lon_"+str(i)+", target_lat_"+str(i)+" = RECC.patch_match(i, edges_"+str(i)+", gt_"+str(i)+", fact_x_"+str(i)+", fact_y_"+str(i)+", x_b_"+str(i)+", y_b_"+str(i)+", mask_"+str(i)+", edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, s, it_cancel, it_max)")
        exec("gcplist, dist_a_"+str(i)+", dist_lon_a_"+str(i)+", dist_lat_a_"+str(i)+", origin_x_a_"+str(i)+", origin_y_a_"+str(i)+", target_lon_a_"+str(i)+", target_lat_a_"+str(i)+" = RECC.remove_outliers(i, ppp, steps, outlier_type, dist_"+str(i)+", dist_lon_"+str(i)+", dist_lat_"+str(i)+", origin_x_"+str(i)+", origin_y_"+str(i)+", target_lon_"+str(i)+", target_lat_"+str(i)+")")
        exec("RECC.georeference(i,wdir,ppp,path[i],files[i],steps,gcplist)")
print("[100%] Done.")   

#%%
#import numpy as np
#import matplotlib.pyplot as plt
#clist = list(np.random.choice(range(256), size=len(dist_1)))
#clist_a = list(np.random.choice(range(256), size=len(dist_a_1)))

#%% TARGET  
#plt.imshow(edges_0, cmap='gray', interpolation='nearest')
#plt.scatter((target_lon_1-gt_1[0])/(gt_1[1]*fact_x_1), (target_lat_1-gt_1[3])/(gt_1[5]*fact_y_1),c=clist)

#%% ORIGIN IN IMAGE
#plt.imshow(edges_1, cmap='gray', interpolation='nearest')
#plt.scatter(origin_x_1/fact_x_1, origin_y_1/fact_y_1,c=clist)

#%% (OUTLIERS) TARGET
#plt.imshow(edges_0, cmap='gray', interpolation='nearest')
#plt.scatter((target_lon_a_1-gt_1[0])/(gt_1[1]*fact_x_1), (target_lat_a_1-gt_1[3])/(gt_1[5]*fact_y_1),c=clist_a)

#%% (OUTLIERS) ORIGIN IN IMAGE
#plt.imshow(edges_1, cmap='gray', interpolation='nearest')
#plt.scatter(origin_x_a_1/fact_x_1, origin_y_a_1/fact_y_1,c=clist_a)
