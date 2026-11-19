import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt,pi
from matplotlib.ticker import FormatStrFormatter
import os, sys
from tensorflow import keras
from tensorflow.keras import layers
from pandas import DataFrame
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
from sklearn.preprocessing import PowerTransformer
from random import *

dir = os.path.dirname(os.path.realpath('__file__'))

###################################################################################################################################


def read_data(data):
    return pd.read_csv(data)


def percentage_normalized_mse(y_true, y_pred):
    nmse_p = 100*(K.sum((y_true-y_pred)**2)/K.sum((y_true)**2))
        
    return nmse_p 



def plot_history(history):
          
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure(figsize=(8,6))
    plt.xlabel("Epochs",fontsize=20)
    plt.ylabel(" Mean Squared Error (Normalized) [$\%$] ",fontsize=20)
    #plt.ylim([0,100])
    plt.plot(hist['epoch'], hist['percentage_normalized_mse'],'b--',
           label='Training Error',color='b')
    plt.plot(hist['epoch'], hist['val_percentage_normalized_mse'],'r',
           label='Validation Error',color='r')
    
    
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('history.png')
    plt.close()
    #hist.to_csv('/history.csv' , index = None , header = True)

########################################## ## Training LSTM model for a given number of increments
    
N_Data = 10000
timesteps = 100

Inputs_name = ['Exx', 'Eyy','Ezz','Exy','Exz','Eyz']
Labels_name = ['C11','C22','C33','C44','C55','C66','C12_m','C13_m','C14_m','C15_m','C16_m','C23_m','C24_m','C25_m','C26_m','C34_m','C35_m','C36_m','C45_m','C46_m','C56_m']




 

Nx = len(Inputs_name)
Ny = len(Labels_name)


scalingX = MinMaxScaler()
scalingY = MinMaxScaler()

raw_dataset1 = DataFrame()
raw_dataset2 = DataFrame()
raw_dataset3 = DataFrame()
raw_dataset4 = DataFrame()
raw_dataset5 = DataFrame()
raw_dataset6 = DataFrame()
raw_dataset7 = DataFrame()
raw_dataset8 = DataFrame()
raw_dataset9 = DataFrame()
raw_dataset10 = DataFrame()

for i in range(1000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset1 = raw_dataset1.append(df)

for i in range(1000,2000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset2 = raw_dataset2.append(df)    
   
for i in range(2000,3000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset3 = raw_dataset3.append(df)

for i in range(3000,4000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset4 = raw_dataset4.append(df)      
   
for i in range(4000,5000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset5 = raw_dataset5.append(df)
   
for i in range(5000,6000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset6 = raw_dataset6.append(df)

for i in range(6000,7000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset7 = raw_dataset7.append(df)    
   
for i in range(7000,8000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset8 = raw_dataset8.append(df)

for i in range(8000,9000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset9 = raw_dataset9.append(df)      
   
for i in range(9000,10000):
    df = read_data('/scratch/adanoun/Param_Analysis_Homogenization_V4/Data_Base/Data_'+str(i)+'.csv')
    raw_dataset10 = raw_dataset10.append(df)    

raw_dataset = pd.concat([raw_dataset1 , raw_dataset2, raw_dataset3 , raw_dataset4, raw_dataset5 ,
                         raw_dataset6 , raw_dataset7, raw_dataset8 , raw_dataset9, raw_dataset10],ignore_index=True)  



   
X = raw_dataset[Inputs_name]
Y = raw_dataset[Labels_name]

X = raw_dataset[Inputs_name]
Y = raw_dataset[Labels_name]




p_train = 0.9
p_test = 0.1

input_shape = len(Inputs_name)
output_shape = len(Labels_name)

N_train = int(p_train*N_Data)
N_test = int(p_test*N_Data)

X_norm = scalingX.fit_transform(X)
X_norm = X_norm.reshape(N_Data,timesteps,Nx)


Y_norm = scalingY.fit_transform(Y)
Y_norm = Y_norm.reshape(N_Data,timesteps,Ny)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=p_test, shuffle=False)

X_train_norm = scalingX.transform(X_train)
Y_train_norm = scalingY.transform(Y_train)
X_test_norm = scalingX.transform(X_test)
Y_test_norm = scalingY.transform(Y_test)


X_train_norm = X_train_norm.reshape(N_train,timesteps,Nx)
Y_train_norm = Y_train_norm.reshape(N_train,timesteps,Ny)

X_test_norm = X_test_norm.reshape(N_test,timesteps,Nx)
Y_test_norm = Y_test_norm.reshape(N_test,timesteps,Ny)    




    
    


    







checkpoint_path = "training_checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model=Sequential()
model.add(LSTM(50,input_shape=(timesteps, Nx), return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dense(50, activation='relu'))
model.add(Dense(Ny, activation='linear'))
model.compile(loss='mse',optimizer='Adam',metrics=[percentage_normalized_mse])
model.summary()

history = model.fit(X_train_norm,Y_train_norm,epochs=1000,validation_split = 0.2,verbose=1,callbacks=[cp_callback], shuffle=True)
hist = DataFrame(history.history)

#lot_history(history)

hist.to_csv('history.csv' , index = None , header = True)


argmin = hist['val_percentage_normalized_mse'].argmin()+1

model=Sequential()
model.add(LSTM(50,input_shape=(timesteps, Nx), return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dense(50, activation='relu'))
model.add(Dense(Ny, activation='linear'))
model.compile(loss='mse',optimizer='Adam',metrics=[percentage_normalized_mse])
model.summary()

model.load_weights('training_checkpoint/cp-0'+str(argmin)+'.ckpt')


#model.load_weights('training/cp-0337.ckpt')


Y_pred_norm = model.predict(X_test_norm)
Y_pred_norm_copy = Y_pred_norm.reshape(N_test*timesteps,Ny)
Y_pred = scalingY.inverse_transform(Y_pred_norm_copy)

#Y_p = pd.DataFrame({Labels_name[i] : Y_p[:,i] for i in range(output_shape)})
Y_pred = pd.DataFrame({Labels_name[i] : Y_pred[:,i] for i in range(output_shape)})

X_test.to_csv('X_test.csv', index=None, header=True)
Y_test.to_csv('Y_test.csv', index=None, header=True)
Y_pred.to_csv('Y_pred.csv', index=None, header=True)





for i in range(2):
    units = 50
    W = model.layers[i].get_weights()[0]
    U = model.layers[i].get_weights()[1]
    b = model.layers[i].get_weights()[2]

    W_i = W[:, :units]
    W_f = W[:, units: units * 2]
    W_c = W[:, units * 2: units * 3]
    W_o = W[:, units * 3:]

    U_i = U[:, :units]
    U_f = U[:, units: units * 2]
    U_c = U[:, units * 2: units * 3]
    U_o = U[:, units * 3:]

    b_i = b[:units]
    b_f = b[units: units * 2]
    b_c = b[units * 2: units * 3]
    b_o = b[units * 3:]


    dic = {}
    dic['W_i'] = W_i
    dic['U_i'] = U_i
    dic['b_i'] = b_i
    dic['W_f'] = W_f
    dic['U_f'] = U_f
    dic['b_f'] = b_f
    dic['W_c'] = W_c
    dic['U_c'] = U_c
    dic['b_c'] = b_c
    dic['W_o'] = W_o
    dic['U_o'] = U_o
    dic['b_o'] = b_o   
    np.savez('LSTM_layer_'+str(i+1),**dic)

for i in range(2,4):
    dic = {}
    dic['W'] = model.layers[i].get_weights()[0]
    dic['b'] = model.layers[i].get_weights()[1]
    np.savez('Dense_layer_'+str(i+1),**dic)


dic = {}
dic['scalingX_min'] = scalingX.data_min_
dic['scalingX_max'] = scalingX.data_max_
dic['scalingY_min'] = scalingY.data_min_
dic['scalingY_max'] = scalingY.data_max_
np.savez('scaling_Lt',**dic)















"""
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model=Sequential()
model.add(LSTM(50,input_shape=(timesteps, Nx), return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dense(50, activation='tanh'))
model.add(Dense(Ny, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=[percentage_normalized_mse])
model.summary()




history = model.fit(X_train_norm,Y_train_norm,epochs=1000,validation_split = 0.2,verbose=1,callbacks=[cp_callback])
hist = DataFrame(history.history)
plot_history(history)

plot_history(history)


#!mkdir -p saved_model
model.save('saved_model/my_model')
model.save('my_model.h5')
#new_model = tf.keras.models.load_model('saved_model/my_model')
#new_model = tf.keras.models.load_model('my_model.h5')
"""
