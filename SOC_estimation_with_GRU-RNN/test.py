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
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Conv1D,AvgPool1D,Concatenate
from keras.layers import LSTM,GRU
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K

import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.io as sio
import numpy as np
import time

#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

# set para
opt_timestep=1000
opt_hiddennode=1000   #
opt_epoch=100
opt_batchsize=72
opt_loss='mae'   #'mse''mape''msle',squared_hinge,categorical_hinge,binary_crossentropy
opt_optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                     #SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
                     # RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
                      #(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                      #Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                      #Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
                      #RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
opt_sampling=1
model = Sequential()            
model.add(GRU(opt_hiddennode, input_shape=(opt_timestep, 6)))
model.add(Dense(50,kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))
model.summary()


#model.compile(loss=opt_loss, optimizer=opt_optimizer)
#history = model.fit(train_X, train_y, epochs=opt_epoch, batch_size=opt_batchsize, validation_data=(test_X, test_y), verbose=2, shuffle=False)

savename= str(opt_timestep)+'Tstep_'+'50Sampl'+str(opt_hiddennode)+'Hnode_'+str(opt_epoch)+'Epoch_'+str(opt_batchsize)+'Bsize_'+str(opt_loss)

h5_name = 'trainmodel/'+savename+'.h5'

model.load_weights(h5_name)

testrootdir = 'testset'
list = os.listdir(testrootdir)
mae_list=[]
for i in range(0,len(list)):
    

    path = os.path.join(testrootdir, list[i])
    if os.path.isfile(path):
        load_data = sio.loadmat(path)
    features = load_data['para']
    dfdata = pd.DataFrame(features)
    values1 = dfdata.values
    tests = values1.astype('float32')

    testshape=(int(tests.shape[0]/opt_timestep)-1)*opt_timestep
    test_X, test_y = tests[:testshape, :-1], tests[:testshape, -1]
# reshape input to be 3D [samples, timesteps, features]

    testingshape = int(testshape/opt_sampling)

    test_Xs=np.random.standard_normal(size=(testingshape, opt_timestep, test_X.shape[1]))
    test_Ys=np.random.standard_normal(size=(testingshape))

    for j in range(0, int(testshape / opt_sampling)):
        test1 = tests[j * opt_sampling:j * opt_sampling + opt_timestep, :-1]
        test_temp = test1.reshape((1, opt_timestep, test_X.shape[1]))
        test_Xs[j, 0:opt_timestep, :] = test_temp
        test_Ys[j] = tests[j * opt_sampling+opt_timestep,-1]

    test_X =  test_Xs
    test_y = test_Ys
    print(test_X.shape, test_y.shape)

    start=time.clock()
    # make a prediction
    pre_x = model.predict(test_X,batch_size=opt_batchsize)
    # calculate RMSE
    mae = mean_absolute_error(pre_x, test_y)
    end=time.clock()

    print('Test MAE: %.3f' % mae)
    print('Runing Time: %.3f s' % (end-start))


    plt.figure(figsize=(10, 6))

    plt.plot(range(len(pre_x)),pre_x,label='pred')
    plt.plot(range(len(test_y)),test_y,label='test_y')
    plt.legend()

    image_name='testresult/'+list[i]+'_'+savename+'_'+'RE'+str(mae)+'_'+'RT'+str(end-start)+'.png'
    plt.savefig(image_name, dpi=200)

    mat_name='testresult/'+list[i]+'_'+savename+'_'+'RE'+str(mae)+'_'+'RT'+str(end-start)+'.mat'
    sio.savemat(mat_name, {'pre_x': pre_x,'test_y': test_y})
    mae_list.append(mae)
print (mae_list)


