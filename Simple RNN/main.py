import MS_Model_LSTM
import MS_Model_RNN
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
#Test RNN
obj = MS_Model_RNN.MS_Model_RNN(n_a=1090,max_val = 0.5)
obj.load_data()
parameters ,a = obj.model(obj.Train_Data_X,obj.Train_Data_Y,num_iterations =151,print_cost=True)

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

obj = MS_Model_LSTM.MS_Model_LSTM(n_a = 32,max_val = 0.001)
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

parameters,a0,c0 = obj.model(obj.Train_X,obj.Train_Y,iterations=151,learning_rate = 0.0001,print_cost = True)

"""
