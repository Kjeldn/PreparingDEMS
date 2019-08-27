import META
import CANNY
import RECC
import RECC2
import numpy as np
import matplotlib.pyplot as plt
import cv2

wdir = r"E:\ORTHODUMP"
files = ["E0","E1"]

path = META.initialize(wdir,files)

temp_path = path[0]
    
img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,temp_path)
edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
edges0C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C         = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)

img_F,img_g_F,img_b_F,mask_b_0,mask_o_0,fact_x_0,fact_y_0,x_b_0,y_b_0,gt_0            = META.switch_correct_ortho(0.5,0.5,temp_path,edgeChainsE_C)

for i in range(1,2):
    print(i)
    temp_path = path[i]
    
    img_C,img_g_C,img_b_C,mask_b_C                                                        = META.correct_ortho(0.5,temp_path)
    edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C  = CANNY.CannyPF(0.5,img_b_C,mask_b_C)
    edgechainmap_C,edgeChainsA_C,edgeChainsB_C,edgeChainsC_C,edgeChainsD_C,edgeChainsE_C  = CANNY.CannyLines(0.5,edgemap_C,gradientMap_C,orientationMap_C,maskMap_C,gradientPoints_C,gradientValues_C)
    exec("edges"+str(i)+"C = edgechainmap_C")
    
    img_F,img_g_F,img_b_F,mask_b_F,mask_o,fact_x,fact_y,x_b,y_b,gt                        = META.switch_correct_ortho(0.5,0.5,temp_path,edgeChainsE_C)
    
#%%    
dst_max = 10
pixel_size = 0.5
w = 50

max_dist = int((dst_max)/pixel_size)
bound1 = mask_o_0.shape[0]
bound2 = mask_o_0.shape[1]
grid = []
for i in range(max_dist+w,bound1-max_dist,2*w):
    for j in range(max_dist+w,bound2-max_dist,2*w):
        if mask_o_0[i,j] == 0:
            grid.append((i,j))
grid = np.array(grid)
target_l = []
patch_l = []
RECC_l = []
o_x = np.zeros(grid.shape)
o_y = np.zeros(grid.shape)
t_x = np.zeros(grid.shape)
t_y = np.zeros(grid.shape)
for i in range(len(grid)):
    x_i_0 = grid[i][0]
    y_i_0 = grid[i][1]
    target = edges0C[x_i_0-w:x_i_0+w,y_i_0-w:y_i_0+w]
    sum_target = np.sum(target)
    search_wide = edges1C[x_i_0-max_dist-w:x_i_0+max_dist+w,y_i_0-max_dist-w:y_i_0+max_dist+w]    
    
    sum_patch = np.zeros((2*max_dist,2*max_dist))
    for x in range(2*max_dist):
        for y in range(2*max_dist):
            patch = search_wide[x-w+w:x+w+w,y-w+w:y+w+w]
            sum_patch[x,y] = np.sum(patch)
                
    numerator_old = np.zeros((2*max_dist,2*max_dist))       
    for x in range(2*max_dist):
        for y in range(2*max_dist):
            patch = search_wide[x-w+w:x+w+w,y-w+w:y+w+w]
            numerator_old[x,y] = np.sum(patch*target) 
            
    numerator_new = np.zeros((2*max_dist,2*max_dist))
    for x in range(2*max_dist):
        for y in range(2*max_dist):
            patch = search_wide[x:x+2*w,y:y+2*w]
            #numerator_new[x,y] = np.fft.irfft2(np.fft.rfft2(patch)*np.fft.rfft2(target,patch.shape))      
            #numerator_new[x,y] = np.fft.irfft(np.dot(np.fft.rfft(np.ravel(patch)),np.fft.rfft(np.rav(target)))) 
            #numerator_new[x,y] = np.fft.ifft(np.dot(np.fft.fft(np.ravel(patch)),np.fft.fft(np.ravel(target))))
    test1 = np.fft.irfft2(np.fft.rfft2(search_wide)*np.fft.rfft2(target,search_wide.shape))
    test2 = np.fft.irfft2(np.fft.rfft2(target)*np.fft.rfft2(search_wide,target.shape)) 
    test3 = np.fft.ihfft(np.fft.hfft(target)[:2*w,:4*w]*np.fft.hfft(search_wide)[:2*w,:4*w-2])
    
    test6 = cv2.filter2D(search_wide,-1,target)
    test6 = test6[w:-w,w:-w]

        
    denominator = sum_target + sum_patch
    search_wide2 = search_wide[:-w,:-w]
    numerator = np.fft.irfft2(np.fft.rfft2(search_wide)*np.fft.rfft2(target,search_wide.shape))
    RECC = numerator/denominator
    RECC_s.fill(np.NaN)
    RECC_s[(x_i_0-max_dist)+1:(x_i_0-max_dist)+1+2*max_dist,(y_i_0-max_dist):(y_i_0-max_dist)+2*max_dist] = RECC[w:,w:]
    RECC_l.append(RECC_s)
    max_one  = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-1)[-1]
    max_n    = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-4-1)[-4-1]
    y_i      = np.where(RECC_s >= max_one)[1][0]  
    x_i      = np.where(RECC_s >= max_one)[0][0]
    y_n      = np.where(RECC_s >= max_n)[1][0:-1]
    x_n      = np.where(RECC_s >= max_n)[0][0:-1]
    target_l.append(target)
    patch_l.append(edges1C[x_i-w:x_i+w,y_i-w:y_i+w])  
    o_x[i] = x_i
    o_y[i] = y_i
    t_x[i] = x_i_0
    t_y[i] = y_i_0   
    
    
    
    
#%%
plt.imshow(edges0C,cmap='gray')  
index = np.where(cv<=20)[0]
plt.scatter(t_y[index],t_x[index],c='r')
plt.scatter(o_y[index],o_x[index],c='b')