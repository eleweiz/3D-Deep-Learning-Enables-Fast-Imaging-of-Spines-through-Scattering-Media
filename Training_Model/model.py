import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras

def unet(pretrained_weights = None,input_size = (64,64,8,1), learning_rate = 1e-4):
    inputs = Input(input_size) 
    # provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 128, 1) for 128x128x128 volumes with a single channel
    # default:"channels_last" corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)
    conv1 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    conv1 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    conv1 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    conv2 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
    conv3 = Conv3D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    conv3 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool3)
    conv4 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 =Conv3D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    conv4 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    # drop4 = Dropout(0.5)(conv4)
    drop4 = conv4
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool4)
    conv5 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv5)
    conv5 = Conv3D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv5 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv5)
    # drop5 = Dropout(0.5)(conv5)
    drop5 = conv5

    # up6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6=Conv3DTranspose(512, 2, strides=(2, 2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4,up6], axis = -1)
    conv6 = Conv3D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge6)
    conv6 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv6)
    conv6 = Conv3D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    conv6 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv6)

    up7=Conv3DTranspose(256, 2, strides=(2, 2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(conv6)
    # up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = -1)
    conv7 = Conv3D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge7)
    conv7 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
    conv7 = Conv3D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    conv7 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
    
    up8=Conv3DTranspose(128, 2, strides=(2, 2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2,up8], axis = -1)
    conv8 = Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge8)
    conv8 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
    conv8 = Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    conv8 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)

    up9=Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1,up9], axis = -1)
    conv9 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge9)
    conv9 =BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
    conv9 = Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(1, 3, padding='same', activation=None, kernel_initializer='he_normal')(conv9)
    added = Add()([conv9, inputs])    
    model = Model(input = inputs, output = added)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # model.compile(optimizer = Adam(lr = learning_rate), loss = 'mean_squared_logarithmic_error', metrics = ['accuracy'])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

'''
Conv2D输出计算:
输入图片矩阵为：WxW
卷积核大小，kernel_size：FxF
步长strides：S
边界扩充padding的值：P

输出大小N：N=(W−F+2P)/S+1

Conv2DTranspose输出计算:
输入图片矩阵为：NxN
卷积核大小，kernel_size：FxF
步长strides：S
边界扩充padding的值：P

输出大小W：W=(N−1)∗S−2P+F

举个栗子，原大小为 X X X，FCN5层池化后为 X/32 X/32X/32，
可以使用下式恢复原来大小：(X/32−1)∗32+32 (X/32-1)*32+32 (X/32−1)∗32+32，
即设卷积核大小和步长为32，padding为0 。
'''

