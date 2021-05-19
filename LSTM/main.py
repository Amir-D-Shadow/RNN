import numpy as np
import os

"""
parameters = obj.initial_parameters(128,49,49)

Waa = parameters["Waa"]
Wax = parameters["Wax"]
Wya = parameters["Wya"]
ba = parameters["ba"]
by = parameters["by"]

dWya = Wya.copy()
dWaa = Waa.copy()
dWax = Wax.copy()
dba =  ba.copy()
dby = by.copy()

gradients = {"dWya":dWya,"dWaa":dWaa,"dWax":dWax,"dba":dba,"dby":dby}
new_grad = obj.gradient_clip(gradients,max_val=0.0001)
"""

"""
#RNN
import MS_Model_RNN

obj = MS_Model_RNN.MS_Model_RNN(n_a=128,max_val = 0.5)
obj.load_data()
parameters ,a = obj.model(obj.Train_Data_X,obj.Train_Data_Y,num_iterations =101,print_cost=True)

Waa = parameters["Waa"]
Wax = parameters["Wax"]
Wya = parameters["Wya"]
ba = parameters["ba"]
by = parameters["by"]

z = np.dot(Wya,a)+by
y_hat,_= obj.sigmoid(z)

result_list = y_hat.flatten().tolist()
res = np.array(result_list).argsort()[-7:]

"""
"""
#Test
obj = MS_Model_LSTM.MS_Model_LSTM(max_val = 0.1)
obj.load_data()

print("X_shape: ",obj.Train_X.shape," ","Y_shape: ",obj.Train_Y.shape)
print("X first row: ",obj.Train_X[0])
print("Y first row: ",obj.Train_Y[0])
print("X last row: ",obj.Train_X[-1])
print("Y last row: ",obj.Train_Y[-1])

np.random.seed(1)
parameters = obj.initialize_parameters(obj.n_a,obj.n_x,obj.n_y)

X = np.random.randn(10,obj.n_x)
Y = np.random.randn(9,obj.n_y)
a0 = np.random.randn(obj.n_a,1)
c0 = np.random.randn(obj.n_a,1)

a,c,cahce,loss = obj.LSTM_forward(X,Y,a0,c0,parameters)
"""
"""
#With peephole connection LSTM
import MS_Model_LSTM

obj = MS_Model_LSTM.MS_Model_LSTM(n_a=148,max_val = 0.01)
obj.load_data()

parameters,at,ct = obj.model(obj.Train_X,obj.Train_Y,iterations = 1001,learning_rate=0.035,regularization_factor=1,beta1=0.99,beta2=0.999,eplison=1e-8,print_cost=True)
Wya = parameters["Wya"]
by = parameters["by"]

z = np.dot(Wya,at)+by
y_hat= obj.sigmoid(z)

res = y_hat.flatten().argsort()[-7:]

x_t = obj.Train_X[-1,:].reshape(obj.n_x,1)
next_res = obj.predict(at,ct,x_t,parameters)
"""

"""
#Without peephole connection LSTM relu
import MS_Model_LSTM_relu as rnn

obj = rnn.MS_Model_LSTM(n_a=312,max_val=0.01)
obj.load_data()

parameters,at,ct = obj.model(obj.Train_X,obj.Train_Y,iterations = 501,learning_rate=0.0035,regularization_factor=1,beta1=0.9,beta2=0.999,eplison=1e-8,print_cost=True)

Wya = parameters["Wya"]
by = parameters["by"]

z = np.dot(Wya,at)+by
z = np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2))
y_hat = obj.sigmoid(z)

res = y_hat.flatten().argsort()[-7:]
"""

"""
#Without peephole connection LSTM tanh
import MS_Model_LSTM_tanh as rnn

obj = rnn.MS_Model_LSTM(n_a=256,max_val=0.01)
obj.load_data()

parameters,at,ct = obj.model(obj.Train_X,obj.Train_Y,iterations = 701,learning_rate=0.0035,regularization_factor=1,beta1=0.9,beta2=0.999,eplison=1e-8,print_cost=True)

Wya = parameters["Wya"]
by = parameters["by"]

z = np.dot(Wya,at)+by
z = np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2))
y_hat = obj.sigmoid(z)

res = y_hat.flatten().argsort()[-7:]

"""
#With peephole connection LSTM (3 gates)
import MS_Model_LSTM_ph_3_gate

obj = MS_Model_LSTM_ph_3_gate.MS_Model_LSTM(n_a=158,max_val = 0.01)
obj.load_data()

parameters,at,ct = obj.model(obj.Train_X,obj.Train_Y,iterations = 901,learning_rate=0.035,regularization_factor=1,beta1=0.99,beta2=0.999,eplison=1e-8,print_cost=True)
Wya = parameters["Wya"]
by = parameters["by"]

z = np.dot(Wya,at)+by
y_hat= obj.sigmoid(z)

res = y_hat.flatten().argsort()[-7:]

x_t = obj.Train_X[-1,:].reshape(obj.n_x,1)
next_res = obj.predict(at,ct,x_t,parameters)
