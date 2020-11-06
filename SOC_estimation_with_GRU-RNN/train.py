from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.io as sio
import numpy as np
import time
#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')


# set para
opt_timestep=1000	#1000,500,250,125
opt_hiddennode=1000   #1000,500,250,125
opt_epoch=100        #20,40,60,80,100,120,140,160,180，200
opt_batchsize=72
opt_loss='mae'   #'mse''mape''msle',squared_hinge,categorical_hinge,binary_crossentropy
opt_optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                     #SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
                     # RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
                      #(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                      #Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                      #Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
                      #RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
opt_sampling=50  #1,250,

model = Sequential()
model.add(GRU(opt_hiddennode, input_shape=(opt_timestep, 6)))
model.add(Dense(50,kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))
model.summary()

savename= str(opt_timestep)+'Tstep_'+str(opt_sampling)+'Sampl'+str(opt_hiddennode)+'Hnode_'+str(opt_epoch)+'Epoch_'+str(opt_batchsize)+'Bsize_'+str(opt_loss)

plot_model(model, to_file=savename+'_model.png', show_shapes=True)


trainrootdir = 'trainset'
list = os.listdir(trainrootdir)
mae_list=[]
Finaltrain_X=[]
Finaltrain_y=[]


for j in range(0,1):
    path = os.path.join(trainrootdir, list[j])
    if os.path.isfile(path):
        load_data = sio.loadmat(path)
    features = load_data['para']
    dfdata = pd.DataFrame(features)
    values1 = dfdata.values
    trains = values1.astype('float32')

    trainshape=(int(trains.shape[0]/opt_timestep)-1)*opt_timestep
    train_X, train_y = trains[:trainshape, :-1], trains[:trainshape, -1]
# reshape input to be 3D [samples, timesteps, features]

    trainingshape = int(trainshape/opt_sampling)

    train_Xs=np.random.standard_normal(size=(trainingshape, opt_timestep, train_X.shape[1]))
    train_Ys=np.random.standard_normal(size=(trainingshape))

    for i in range(0, int(trainshape / opt_sampling)):
        train1 = trains[i * opt_sampling:i * opt_sampling + opt_timestep, :-1]
        train_temp = train1.reshape((1, opt_timestep, train_X.shape[1]))
        train_Xs[i, 0:opt_timestep, :] = train_temp
        train_Ys[i] = trains[i*opt_sampling+opt_timestep,-1]

    train_X =  train_Xs
    train_y = train_Ys
    print(train_X.shape, train_y.shape)
    print('Preprocessing the %.3f th trainset' % j )

Finaltrain_X=train_X
Finaltrain_y=train_y


for j in range(1,len(list)):
    path = os.path.join(trainrootdir, list[j])
    if os.path.isfile(path):
        load_data = sio.loadmat(path)
    features = load_data['para']
    dfdata = pd.DataFrame(features)
    values1 = dfdata.values
    trains = values1.astype('float32')

    trainshape=(int(trains.shape[0]/opt_timestep)-1)*opt_timestep
    train_X, train_y = trains[:trainshape, :-1], trains[:trainshape, -1]
# reshape input to be 3D [samples, timesteps, features]

    trainingshape = int(trainshape/opt_sampling)

    train_Xs=np.random.standard_normal(size=(trainingshape, opt_timestep, train_X.shape[1]))
    train_Ys=np.random.standard_normal(size=(trainingshape))

    for i in range(0, int(trainshape/opt_sampling)):
        train1 = trains[i * opt_sampling:i * opt_sampling + opt_timestep, :-1]
        train_temp = train1.reshape((1, opt_timestep, train_X.shape[1]))
        train_Xs[i, 0:opt_timestep, :] = train_temp
        train_Ys[i] = trains[i*opt_sampling+opt_timestep,-1]

    train_X =  train_Xs
    train_y = train_Ys
    print(train_X.shape, train_y.shape)
    #print('trainingset: %.3f' % i)

    Finaltrain_X=np.concatenate((Finaltrain_X,train_X),axis=0)
    Finaltrain_y = np.concatenate((Finaltrain_y, train_y), axis=0)
    print('Preprocessing the %.3f th trainset' % j )

   #shuffle the trainingdata
np.random.seed(Finaltrain_X.shape[0])
index = [i for i in range(Finaltrain_X.shape[0])]
Finaltrain_Xs=np.random.standard_normal(size=(Finaltrain_X.shape[0], Finaltrain_X.shape[1], Finaltrain_X.shape[2]))
Finaltrain_ys=np.random.standard_normal(size=(Finaltrain_X.shape[0]))
np.random.shuffle(index)
for k in range(0,Finaltrain_X.shape[0]):
    Finaltrain_Xs[k,:,:]=Finaltrain_X[index[k],:,:]
    Finaltrain_ys[k]= Finaltrain_y[index[k]]

   #Training
print('Get final trainingset!! Staring training!')
start=time.clock()
model.compile(loss=opt_loss, optimizer=opt_optimizer)
history = model.fit(Finaltrain_Xs, Finaltrain_ys, epochs=opt_epoch, batch_size=opt_batchsize,validation_split=0.2, validation_data=None,
                        verbose=2, shuffle=True)


end=time.clock()
print('Train Time: %.3f s' % (end-start))

h5_name = 'trainmodel/'+savename+'.h5'
model.save_weights(h5_name)

mat_name='trainmodel/'+savename+'_traintime_'+str(end-start)+'.mat'
sio.savemat(mat_name, {'loss': history.history['loss'],'val_loss': history.history['val_loss']})




