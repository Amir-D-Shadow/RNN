import numpy as np
import os


class Dense_layer:


    def __init__(self,activation,backward_activation,layer_dims=[10,5]):

        """
        layer_dims : list -- indicating how many hidden units in each layer (**notice that layer_dims[0] is dimensions of input layer not hidden layer)
        activation -- list -- a list containing functions (**notice that activation[0] is the activation function of first hidden layer)
        backward_activation -- a list containing function that calcualte derivative of activation of corresponding layer 
        """

        self.layer_dims = layer_dims[:]
        self.activation = activation[:]
        self.backward_activation = backward_activation[:]


    def initialize_parameters(self):

        length = len(self.layer_dims)

        parameters = {}

        for l in range(1,length):

            para_W = "W" + str(l)
            para_b = "b" + str(l)

            parameters[para_W] = np.random.randn(layers[l],layers[l-1])
            parameters[para_b] = np.zeros((layers[l],1))

        return parameters

    def initialize_Adam(self,parameters):

        v = {}
        s = {}

        for para in parameters.keys():

            grad = "d" + para

            v[grad] = np.zeros_like(parameters[para])
            s[grad] = np.zeros_like(parameters[para])

        return v,s

    def update_parameters_with_Adam(self,parameters,gradients,v,s,L,learning_rate=0.01,beta1=0.9,beta2=0.999,eplison=1e-8):

        v_corrected = {}
        s_corrected = {}

        for para in parameters.keys():

            grad = "d"+para

            v[grad] = beta1*v[grad]+(1-beta1)*gradients[grad]
            s[grad] = beta2*s[grad]+(1-beta2)*(gradients[grad]**2)

            v_corrected = v[grad]/(1-beta1**L)
            s_corrected = s[grad]/(1-beta2**L)

            parameters[para] -= learning_rate*v_corrected/np.sqrt(s_corrected+eplison)

        return parameters,v,s


    def step_forward(self,a_prev,WL,bL,act_func_L):

        z = np.dot(WL,a_prev)+bL
        a_next = act_func_L(z)

        cache_L = (a_next,a_prev,z,WL,bL)

        return a_next,cache_L

    def forward_propagation(self,a0,parameters):

        length = len(self.layer_dims)

        a_prev = a0

        a = []
        cache = []

        for l in range(1,length):

            a_prev,cache_L = self.step_forward(a_prev,self.parameters["W"+str(l)],parameters["b"+str(l)],activation[l-1])
            
            a.append(a_prev)
            cache.append(cache_L)

        return a,cache


    def step_backward(self,da_next,backward_activation_L,cache_L):

        """
        cache_L: (a_next,a_prev,z,WL,bL)
        """
        a_next,a_prev,z,WL,bL = cache_L
        
        dZ = backward_activation_L(z) * da_next
        dW = np.dot(dZ,a_prev.T)
        db = dZ
        da_prev = np.dot(WL.T,dZ)

        return da_prev,dW,db


    def back_propagation(self,dAL,cache,parameters):

        length = len(self.layer_dims)
        da_next = dAL

        gradients = {}

        for l in reversed(range(1,length)):

            grad_W = "dW"+str(l)
            grad_b = "db"+str(l)
            grad_A = "dA"+str(l)

            da_next,dW,db = self.step_backward(da_next,self.backward_activation[l-1],cache_L)

            gradients[grad_A] = da_next
            gradients[grad_W] = dW
            gradients[grad_b] = db

        return gradients

    def optimize(self,a0,parameters,v,s,L,learning_rate=0.01,beta1=0.9,beta2=0.999,eplison=1e-8):

        #forward propogation
        a,cache = self.forward_propagation(a0,parameters)

        #calculate dAL
        

        #backward propogation
        gradients = self.back_propagation(dAL,cache,parameters)

        #update parameters
        parameters,v,s = self.update_parameters_with_Adam(parameters,gradients,v,s,L,learning_rate=0.01,beta1=0.9,beta2=0.999,eplison=1e-8)

        return parameters,a[-1],v,s


    def model(self,X,Y,iterations=100,learning_rate=0.01,beta1=0.9,beta2=0.999,eplison=1e-8):

        parameters = self.initialize_parameters()
        v,s = self.initialize_Adam(parameters)

        #calculate first dAL

        for i in range(iterations):

            parameters,aL,v,s = self.optimize(a0,parameters,v,s,L,learning_rate=0.01,beta1=0.9,beta2=0.999,eplison=1e-8)

        return parameters,aL




    
        
        

    


        
