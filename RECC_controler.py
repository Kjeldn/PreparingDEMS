#%% INPUT
wdir = r"E:"
files = ["0","1"]

file_type      = 0         # [0: Orthomosaic] [1: DEM]
outlier_type   = 1         # [0: Boxplot] [1: 2D Confidence]
conf           = 75        # {50,75,80,90,95}
gamma_correct  = 1         # {0,1}
extra_blur     = 1         # {0,1}
luma           = 709       # {240,601,709}
pixel_size     = 0.6       # One pixel will be <...> m by <...> m.
ppp            = 25        # Use <...> points per photo to georeference.
cv_max         = 40        # Approve matches only if CV score is under {<...>} .
dst_max        = 20        # Approve matches only if distance is under {<...>} [m] .
w              = 30        # Search square will have a length of {2*<...>*pixel_size} [m].
v              = 30        # Masking square will have a length of {2*<...> pixel_size} [m].
s              = 4         # RECC search range distance is capped at {<...>*w*pixel_size} [m].
it_cancel      = 1         # {0,1} 
it_max         = 5         # Max number of iterations per match

import RECC

#%%
path,steps = RECC.initialize(wdir,files,ppp)
for i in range(len(path)):
    exec("edges_"+str(i)+", gt_"+str(i)+", fact_x_"+str(i)+", fact_y_"+str(i)+", x_b_"+str(i)+", y_b_"+str(i)+", mask_"+str(i)+" = RECC.file_to_edges(i,file_type,path[i],luma,gamma_correct,pixel_size,extra_blur,ppp,steps)")
    if i >= 1:
        exec("dist_"+str(i)+", dist_lon_"+str(i)+", dist_lat_"+str(i)+", origin_x_"+str(i)+", origin_y_"+str(i)+", target_lon_"+str(i)+", target_lat_"+str(i)+", o_x_"+str(i)+", o_y_"+str(i)+", t_x_"+str(i)+", t_y_"+str(i)+" = RECC.patch_match(i, edges_"+str(i)+", gt_"+str(i)+", fact_x_"+str(i)+", fact_y_"+str(i)+", x_b_"+str(i)+", y_b_"+str(i)+", mask_"+str(i)+", edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, s, it_cancel, it_max)")
        exec("gcplist, dist_a_"+str(i)+", dist_lon_a_"+str(i)+", dist_lat_a_"+str(i)+", origin_x_a_"+str(i)+", origin_y_a_"+str(i)+", target_lon_a_"+str(i)+", target_lat_a_"+str(i)+",o_x_a_"+str(i)+", o_y_a_"+str(i)+", t_x_a_"+str(i)+", t_y_a_"+str(i)+" = RECC.remove_outliers(i, ppp, conf, steps, outlier_type, dist_"+str(i)+", dist_lon_"+str(i)+", dist_lat_"+str(i)+", origin_x_"+str(i)+", origin_y_"+str(i)+", target_lon_"+str(i)+", target_lat_"+str(i)+",o_x_"+str(i)+", o_y_"+str(i)+", t_x_"+str(i)+", t_y_"+str(i)+")")
        #exec("RECC.georeference(i,wdir,ppp,path[i],files[i],steps,gcplist)")
print("[100%] Done.")        

#%%
#import numpy as np
#import matplotlib.pyplot as plt
#clist = list(np.random.choice(range(256), size=len(dist_1)))
#clist_a = list(np.random.choice(range(256), size=len(dist_a_1)))

#%% TARGET  
#plt.imshow(edges_0, cmap='gray', interpolation='nearest')
#plt.scatter((target_lon_1-gt_1[0])/(gt_1[1]*fact_x_1), (target_lat_1-gt_1[3])/(gt_1[5]*fact_y_1),c=clist)

#%% ORIGIN
#plt.imshow(edges_1, cmap='gray', interpolation='nearest')
#plt.scatter(origin_x_1/fact_x_1, origin_y_1/fact_y_1,c=clist)

#%% DOUBLE SCATTER
#plt.scatter(o_x_1,o_y_1,c=clist)
#plt.scatter(t_x_1,t_y_1,c=clist)

#%% (INLIERS) TARGET
#plt.imshow(edges_0, cmap='gray', interpolation='nearest')
#plt.scatter((target_lon_a_1-gt_1[0])/(gt_1[1]*fact_x_1), (target_lat_a_1-gt_1[3])/(gt_1[5]*fact_y_1),c=clist_a)

#%% (INLIERS) ORIGIN
#plt.imshow(edges_1, cmap='gray', interpolation='nearest')
#plt.scatter(origin_x_a_1/fact_x_1, origin_y_a_1/fact_y_1,c=clist_a)

#%% (INLIERS) DOUBLE SCATTER
#plt.scatter(o_x_a_1,o_y_a_1,c=clist_a)
#plt.scatter(t_x_a_1,t_y_a_1,c=clist_a)

#%% STORE WORKSPACE
import shelve
my_shelf = shelve.open(wdir+"\\shelf.out",'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

#%% LOAD WORKSPACE
#import shelve
#my_shelf = shelve.open(wdir+"\\shelf.out")
#for key in my_shelf:
#    try:
#        globals()[key]=my_shelf[key]
#    except:
#        print('ERROR restoring: {0}'.format(key))
#my_shelf.close()