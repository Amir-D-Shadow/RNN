import MS_Model_RNN
import numpy as np

obj = MS_Model_RNN.MS_Model_RNN(max_val = 0.08)

obj.load_data()

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

parameters ,a = obj.model(obj.Train_Data_X,obj.Train_Data_Y,num_iterations =1,print_cost=True)

Waa = parameters["Waa"]
Wax = parameters["Wax"]
Wya = parameters["Wya"]
ba = parameters["ba"]
by = parameters["by"]

z = np.dot(Wya,a)+by
y_hat,_= obj.sigmoid(z)

result_list = y_hat.flatten().tolist()
res = np.array(result_list).argsort()[-7:]
