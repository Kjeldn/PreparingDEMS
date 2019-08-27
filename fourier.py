#FOURIER
import numpy as np
import matplotlib.pyplot as plt
w=15

target = np.zeros((2*w,2*w))
target[:,w] = 1
sum_target = np.sum(target)

#%%
search_area =  np.random.rand(50,50)
search_area[search_area>0.5]=1
search_area[search_area<1]=0

search_area[5:5+2*w,5:5+2*w] =0
search_area[5:5+2*w,5+w]=1

#%% RECC OLD
RECC_s = np.zeros(search_area.shape)
RECC_s.fill(np.NaN)
for x in range(w,RECC_s.shape[0]-w):
    for y in range(w,RECC_s.shape[1]-w):
        patch = search_area[x-w:x+w,y-w:y+w]
        RECC_s[x,y]=np.sum(np.multiply(target,patch))/(sum_target+np.sum(patch))      
max_one    = np.partition(RECC_s[~np.isnan(RECC_s)].flatten(),-1)[-1]        
x_i_old    = np.where(RECC_s >= max_one)[1][0]
y_i_old    = np.where(RECC_s >= max_one)[0][0]   

#%% RECC NEW    
RECC = (np.fft.irfft2(np.fft.rfft2(search_area)*np.fft.rfft2(target,search_area.shape)))/(np.fft.irfft2(np.fft.rfft2(search_area)*np.fft.rfft2(np.ones(target.shape),search_area.shape))+sum_target)
max_one  = np.partition(RECC[~np.isnan(RECC)].flatten(),-1)[-1]        
x_i_new    = np.where(RECC >= max_one)[1][0] -w
y_i_new    = np.where(RECC >= max_one)[0][0] -w+1

RECC_a = np.zeros(RECC.shape)
RECC_a[1:-w+1,:-w] = RECC[w:,w:]
max_one  = np.partition(RECC_a[~np.isnan(RECC_a)].flatten(),-1)[-1]        
x_i_newnew    = np.where(RECC_a >= max_one)[1][0]
y_i_newnew    = np.where(RECC_a >= max_one)[0][0]
#%%
#plt.imshow(target)
#plt.scatter(w,w,c='r')
#%%
plt.imshow(search_area)
plt.scatter(x_i_old,y_i_old,c='r')
plt.scatter(x_i_newnew,y_i_newnew,c='b')