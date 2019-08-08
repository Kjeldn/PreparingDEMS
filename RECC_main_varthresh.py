#%% INPUT
wdir = r"E:"
files = ["0","1"]
#wdir = r"E:\190806_DEMs_Boomgaard\Ortho"
#files = ["0","1"]

file_type      = 0         # [0: Orthomosaic] [1: DEM]
outlier_type   = 0         # [0: Boxplot] [1: 99%]
gamma_correct  = 1         # {0,1}
luma           = 709       # {240,601,709}
pixel_size     = 0.6       # One pixel will be <...> m by <...> m.
ppp            = 25        # Use <...> points per photo to georeference.
cv_max         = 40        # Approve matches only if CV score is under <...> .
dst_max        = 20        # Approve matches only if distance is under <...> .
w              = 30        # Patch-match will be 2*<...> pixels by 2*<...> pixels .
v              = 30        # Masking area will be 2*<...> pixels by 2*<...> pixels .
it_cancel      = 0         # {0,1} 
it_max         = 5         # Max number of iterations per match

import RECC
import matplotlib.pyplot as plt

path,steps = RECC.initialize(wdir,files,ppp)

#%% TARGET IMAGE
print("[0%] Getting edges for target image.")
edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0 = RECC.file_to_edges(file_type,path[0],luma,gamma_correct)

#%% IMAGES TO BE MATCHED
for i in range(1,len(path)):
    print("["+"{:.0f}".format((1+(i-1)*(ppp+3))/steps)+"%] Getting edges for image nr "+str(i)+".")
    edges, gt, fact_x, fact_y, x_b, y_b, mask = RECC.file_to_edges(file_type,path[i],luma,gamma_correct)
         
    print("["+"{:.0f}".format((2+(i-1)*(ppp+3))/steps)+"%] Matching patches for image nr "+str(i)+".")
    dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat = RECC.patch_match(i, edges, gt, fact_x, fact_y, x_b, y_b, mask, edges_0, gt_0, fact_x_0, fact_y_0, x_b_0, y_b_0, mask_0, ppp, cv_max, dst_max, w, v, steps, it_cancel, it_max)
    
    gcplist, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat = RECC.remove_outliers(i, ppp, steps, outlier_type, dist, dist_lon, dist_lat, origin_x, origin_y, target_lon, target_lat)
    
    print("["+"{:.0f}".format(((3+ppp)+(i-1)*(ppp+3))/steps)+"%] Georeferencing image nr "+str(i)+".")
    RECC.georeference(wdir,path[i],files[i],gcplist)

print("[100%] Done.")

#%%
plt.imshow(edges, cmap='gray', interpolation='nearest')
plt.scatter(origin_x/fact_x,origin_y/fact_y)

#%%
#plt.imshow(mask_0_c)


