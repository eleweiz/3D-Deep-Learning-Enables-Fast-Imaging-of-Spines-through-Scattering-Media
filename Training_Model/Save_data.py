from model import *
from data import *
from keras.utils import plot_model

import matplotlib.pyplot as plt
import scipy.io as sio 
import h5py
import time
H=128
W=128
D=64
# H=64
# W=64
# D=16
                
image_datagen = ImageDataGenerator()
matfn = './data/Testing_SpineExp_in_T30_P1024.mat'
# data = sio.loadmat(matfn)
data= h5py.File(matfn)
trainX = data['I_in']

matfn = './data/Testing_SpineExp_out_T30_P1024.mat'
data= h5py.File(matfn)
trainY = data['I_out']
trainX=np.reshape(trainX,trainX.shape + (1,)) 
trainY=np.reshape(trainY,trainY.shape + (1,)) 

tem=np.size(trainX,0)
D_tem=np.size(trainX,3)
D_loop=np.int_(np.floor(D_tem/D))
Train=0
a_val=np.arange(Train,tem, 1, int)

model = unet(input_size = (H,W,D,1), learning_rate = 1e-4)

############ test
# '''
model.load_weights("unetS1_Exp3_NoRatio_noise_TM_SL65-SpineN-10-22.17.hdf5")
# model = load_model('my_model.h5') # loading model 
Inx_N=np.int_(np.floor(tem-Train))
I_CNN = np.empty((Inx_N, H, W, D_tem))
# GT_CNN = np.empty((Inx_N, H, W, D_tem))
O_CNN = np.empty((Inx_N, H, W, D_tem))
Results_CNN={}


start = time.clock()
for inx1 in range(Inx_N):   # loop for patch
    inx=a_val[inx1]
    for inx_D in range(D_loop):  # loop for set of Depths
        inx_D_tem=(inx_D)*D
        test_inp=trainX[inx,:,:,inx_D_tem:inx_D_tem+D]
        test_inp=np.reshape(test_inp, (1,)+test_inp.shape) 
        results = model.predict(test_inp,batch_size=1)
        for test_inx_tem in range(D):     # loop for  Depths
            test_inx=inx_D_tem+test_inx_tem
            results_m=np.squeeze(results[0,:,:,test_inx_tem])
            I_CNN[inx1,:,:,test_inx]=np.squeeze(trainX[inx,:,:,test_inx,:])
            # GT_CNN[inx1,:,:,test_inx]=np.squeeze(trainY[inx,:,:,test_inx,:])
            O_CNN[inx1,:,:,test_inx]=np.squeeze(results_m)

elapsed = (time.clock() - start)
print("Time used:",elapsed)
dataNew = './data/ResultsN_SpineExp_TM_P1024_Exp3_NoRatio_noise_SLYFP44_d1_np200.mat'    # save data
Results_CNN={'I_CNN':I_CNN, 'O_CNN':O_CNN}
sio.savemat(dataNew, Results_CNN) 








