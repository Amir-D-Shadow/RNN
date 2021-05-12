import numpy as np
import os

class MS_Model_LSTM:

    def __init__(self,n_a=128,n_x=49,n_y=49,max_val=0.8):

        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.maxValue = max_val

        self.Train_X = None
        self.Train_Y = None

    def load_data(self):

        """
        X,Y: farthest --> latest
        """

        path = os.getcwd() + "/" + "Data" + "/" + "MarkSixData.txt"

        X = np.loadtxt(path,delimiter=",",dtype=int)
        res = []
        
        for row in reversed(range(X.shape[0])):

            vec = np.zeros((self.n_x,1))

            for num in X[row]:
        
                vec[num-1] = 1

            res.append(vec)

        self.Train_X = np.squeeze(np.array(res))
        self.Train_Y = self.Train_X[1:].copy()
        

    def sigmoid(self,z):

        return np.where(z>=0,(1/(1+np.exp(-z))),(np.exp(z)/(1+np.exp(z))))

    def relu(self,z):

        return np.maximum(z,0)

    def gradient_clip(self,gradients,maxValue):

        for grad in gradients.values():

            np.clip(a=grad,a_min=-maxValue,a_max=maxValue,out=grad)

        return gradients

    def initialize_parameters(self,n_a,n_x,n_y):

        """
        Parameters includes:

        c_st:
        
        Wca : (n_a,n_a)
        Wcx : (n_a,n_x)
        bc : (n_a,1)

        Gamma_u:

        Wua : (n_a,n_a)
        Wux : (n_a,n_x)
        Wuc : (n_a,n_a)
        bu : (n_a,1)

        Gamma_f:

        Wfa : (n_a,n_a)
        Wfx : (n_a,n_x)
        Wfc : (n_a,n_a)
        bf : (n_a,1)

        Gamma_o:

        Woa : (n_a,n_a)
        Wox : (n_a,n_x)
        bo : (n_a,1)

        y_hat:

        Wya: (n_y,n_a)
        by: (n_y,1)
        
        """

        parameters = {}

        parameters["Wca"] = np.random.randn(n_a,n_a)
        parameters["Wcx"] = np.random.randn(n_a,n_x)
        parameters["bc"] = np.zeros((n_a,1))

        parameters["Wua"] = np.random.randn(n_a,n_a)
        parameters["Wux"] = np.random.randn(n_a,n_x)
        parameters["Wuc"] = np.random.randn(n_a,n_a)
        parameters["bu"] = np.zeros((n_a,1))

        parameters["Wfa"] = np.random.randn(n_a,n_a)
        parameters["Wfx"] = np.random.randn(n_a,n_x)
        parameters["Wfc"] = np.random.randn(n_a,n_a)
        parameters["bf"] = np.zeros((n_a,1))

        parameters["Woa"] = np.random.randn(n_a,n_a)
        parameters["Wox"] = np.random.randn(n_a,n_x)
        parameters["bo"] = np.zeros((n_a,1))

        parameters["Wya"] = np.random.randn(n_y,n_a)
        parameters["by"] = np.zeros((n_y,1))

        return parameters


    def LSTM_step_forward(self,c_prev,a_prev,x_t,parameters):

        """
        c_prev : (n_a,1)
        a_prev : (n_a,1)
        x_t: (n_x,1)

        """

        Wca = parameters["Wca"]
        Wcx = parameters["Wcx"]
        bc = parameters["bc"]

        Wua = parameters["Wua"]
        Wux = parameters["Wux"]
        Wuc = parameters["Wuc"]
        bu = parameters["bu"]

        Wfa = parameters["Wfa"]
        Wfx = parameters["Wfx"]
        Wfc = parameters["Wfc"]
        bf = parameters["bf"]

        Woa = parameters["Woa"]
        Wox = parameters["Wox"]
        bo = parameters["bo"]

        Wya = parameters["Wya"]
        by = parameters["by"]


        #~C_t
        z = np.dot(Wca,a_prev)+np.dot(Wcx,x_t)+bc
        c_st = np.tanh(z)

        #Update Gate
        z = np.dot(Wua,a_prev)+np.dot(Wux,x_t)+np.dot(Wuc,c_prev)+bu
        Gamma_u = self.sigmoid(z)

        #Forget Gate
        z = np.dot(Wfa,a_prev)+np.dot(Wfx,x_t)+np.dot(Wfc,c_prev)+bf
        Gamma_f = self.sigmoid(z)

        #Output Gate
        z = np.dot(Woa,a_prev)+np.dot(Wox,x_t)+bo
        Gamma_o = self.sigmoid(z)

        #Cells state t
        c_t = Gamma_u*c_st+Gamma_f*c_prev

        #Hidden state t
        a_t = Gamma_o*np.tanh(c_t)

        #y_hat prediction
        z = np.dot(Wya,a_t)+by
        y_hat = self.sigmoid(z)

        cache = (y_hat,a_t,c_t,Gamma_o,Gamma_f,Gamma_u,c_st)

        return a_t,c_t,y_hat,cache


    def LSTM_forward(self,X,Y,a0,c0,parameters):

        """
        X : (T_x,n_x)
        Y : (T,n_y)
        a0: (n_a,1)

        """

        #Get Shape
        T_x,n_x = X.shape
        T,n_y = Y.shape

        #Set up caches
        caches = []
        a = []
        c = []

        #initialize variable
        loss = 0
        a_next = a0.copy()
        c_next = c0.copy()

        for t in range(1,T):

            #Get One Step data X input
            x_t = X[t-1,:].reshape(n_x,1)

            #Forward One Step
            a_next,c_next,y_hat,cache_t = self.LSTM_step_forward(c_next,a_next,x_t,parameters)

            #Save Cell and hidden state
            a.append(a_next)
            c.append(c_next)

            #Update loss
            y_t = Y[t,:].reshape(n_y,1)
            loss = loss + np.sum(- y_t*np.log(y_hat)-(1-y_t)*np.log(1-y_hat))

            #Save Cache
            caches.append(cache_t)


        return a,c,caches,loss
        

        
        

























        
        
