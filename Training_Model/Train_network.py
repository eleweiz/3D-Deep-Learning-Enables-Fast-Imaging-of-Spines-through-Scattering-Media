from model import *
from data import *
from keras.utils import plot_model
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio 
from Generator_data_function import DataGenerator
H=128
W=128
D=64
# H=64
# W=64
# D=16
                    
image_datagen = ImageDataGenerator()
matfn = './data/Training_SpineN_in_TM_rand_Exp3_NoRatio_noise_YFP44_d1_np200.mat'
# data = sio.loadmat(matfn)
data= h5py.File(matfn)
trainX = data['I_in']
trainX=np.reshape(trainX,trainX.shape + (1,)) 
data=None
matfn1 = './data/Training_SpineN_out_TM_rand_Exp3_NoRatio_noise_YFP44_d1_np200.mat'
data1= h5py.File(matfn1)
trainY = data1['I_out']
data1=None
trainY=np.reshape(trainY,trainY.shape + (1,))
print(np.isnan(trainX).any())
print(np.isnan(trainY).any()) 
portion=0.95
tem=np.size(trainX,0)
Train=np.floor(tem*portion)
a_train=np.arange(0, Train, 1, int)
a_val=np.arange(Train,tem, 1, int)

#######  Generator
# Parameters
params = {'dim': (H,W,D),
          'batch_size': 3,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(a_train, trainX, trainY, **params)
validation_generator = DataGenerator(a_val, trainX, trainY, **params)


model = unet(input_size = (H,W,D,1), learning_rate = 1e-4)
plot_model(model, to_file='model_fig.png')
filepath="unetS1_Exp3_NoRatio_noise_TM_SLYFP44_d1_np200-SpineN-{epoch:02d}-{val_loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, period=1)
# period, the distance of epochs for saving points
''' hdf5 包括：
模型的结构，以便重构该模型
模型的权重
训练配置（损失函数，优化器等）
优化器的状态，以便于从上次训练中断的地方开始
(Batchsize, H, W, Channels)
(num_samples, H, W, Channels)
'''
############ train
# '''
history =model.fit_generator(training_generator,steps_per_epoch=300,epochs=18,callbacks=[model_checkpoint],validation_data=validation_generator, validation_steps=20)
 
# steps_per_epoch: typically be equal to ceil(num_samples / batch_size)
# validation_steps: Only relevant if validation_data is a generator. Total number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size.
# An epoch finishes when steps_per_epoch batches have been seen by the model.
# visulization loss
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


dataNew = './data/loss_history_spineN_TM_S1_Exp3_SL65_YFP44_d1_np200_NoRatio_noise.mat'    # save data
Results_CNN=history.history
sio.savemat(dataNew, history.history) 
