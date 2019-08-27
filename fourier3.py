#FOURIER
import numpy as np
import matplotlib.pyplot as plt
w=2

target = np.zeros((2*w,2*w))
target[:,w] = 1
sum_target = np.sum(target)

#%%
search_area =  np.random.rand(50,50)
search_area[search_area>0.5]=1
search_area[search_area<1]=0

search_area[5:5+2*w,5:5+2*w] =0
search_area[5:5+2*w,5+w]=1

#%%
sum_patch = cv2.filter2D(search_area,-1,np.ones(target.shape))
numerator = cv2.filter2D(search_area,-1,target)
RECC = numerator / (sum_patch+sum_target)
RECC_s = RECC[w:-w,w:-w]
max_one  = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-1)[-1]
y_i      = np.where(RECC_s >= max_one)[1][0]  
x_i      = np.where(RECC_s >= max_one)[0][0]


#%%
plt.imshow(search_area)
plt.scatter(x_i+w,y_i+w,c='r')
#%%
plt.imshow(target)
plt.scatter(w,w)