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

    def initialize_Adam(self,gradients):

        #Set up
        v = {}
        s = {}

        for grad in gradients.keys():

            v[grad] = np.zeros_like(gradients[grad])
            s[grad] = np.zeros_like(gradients[grad])

        return v,s

    def update_parameters_with_Adam(self,gradients,parameters,v,s,i,beta1=0.9,beta2=0.999,eplison=1e-8,learning_rate=0.001):

        """
        Adam update parameters per ITERATIONS!!!
        
        ***Important: v,s and v_corrected,s_corrected should be treated separately

        """

        #Important: v,s and v_corrected,s_corrected should be treated separately

        #Set up
        v_corrected = {}
        s_corrected = {}

        #Update parameters
        for para in parameters:

            grad = "d" + para

            #Update v
            v[grad] = beta1*v[grad]+(1-beta1)*gradients[grad]
            v_corrected[grad] = v[grad]/(1-beta1**i)

            #update s
            s[grad] = beta2*s[grad]+(1-beta2)*((gradients[grad])**2)
            s_corrected[grad] = s[grad]/(1-beta2**i)

            #update parameters
            parameters[para] -= learning_rate*(v_corrected[grad]/np.sqrt(s_corrected[grad]+eplison))
            

        return parameters,v,s

    def update_parameters(self,gradients,parameters,learning_rate):

        for para in parameters.keys():

            grad = "d" + para
            parameters[para] -= learning_rate*gradients[grad]

        return parameters
        

    def sigmoid(self,z):

        return np.where(z>=0,(1/(1+np.exp(-z))),(np.exp(z)/(1 + np.exp(z))))

    def relu(self,z):

        return np.maximum(z,0)

    def activation_derivative(self,z,activation="relu"):

        if activation == "tanh":

            return (1-((np.tanh(z))**2))

        elif activation == "relu":

            drelu = np.zeros(z.shape)
            drelu[z>=0] = 1

            return drelu
        
        elif activation == "sigmoid":

            g = self.sigmoid(z)
            
            return g*(1-g)

        else:

            print("Invalid Activation")
            

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
        parameters["bu"] = np.zeros((n_a,1))

        parameters["Wfa"] = np.random.randn(n_a,n_a)
        parameters["Wfx"] = np.random.randn(n_a,n_x)
        parameters["bf"] = np.zeros((n_a,1))

        parameters["Woa"] = np.random.randn(n_a,n_a)
        parameters["Wox"] = np.random.randn(n_a,n_x)
        parameters["bo"] = np.zeros((n_a,1))

        parameters["Wya"] = np.random.randn(n_y,n_a)
        parameters["by"] = np.zeros((n_y,1))

        return parameters


    def LSTM_step_forward(self,a_prev,c_prev,x_t,y_t,parameters):

        """
        c_prev : (n_a,1)
        a_prev : (n_a,1)
        x_t: (n_x,1)
        y_t: (n_y,1)

        """

        Wca = parameters["Wca"]
        Wcx = parameters["Wcx"]
        bc = parameters["bc"]

        Wua = parameters["Wua"]
        Wux = parameters["Wux"]
        bu = parameters["bu"]

        Wfa = parameters["Wfa"]
        Wfx = parameters["Wfx"]
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
        z = np.dot(Wua,a_prev)+np.dot(Wux,x_t)+bu#+np.dot(Wuc,c_prev)+bu
        Gamma_u = self.sigmoid(z)

        #Forget Gate
        z = np.dot(Wfa,a_prev)+np.dot(Wfx,x_t)+bf#+np.dot(Wfc,c_prev)+bf
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


        cache = (y_hat,a_t,c_t,x_t,y_t,a_prev,c_prev,Gamma_o,Gamma_f,Gamma_u,c_st)

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

        for t in range(T):

            #Get One Step data X input
            x_t = X[t,:].reshape(n_x,1)
            y_t = Y[t,:].reshape(n_y,1)
            
            #Forward One Step
            a_next,c_next,y_hat,cache_t = self.LSTM_step_forward(a_next,c_next,x_t,y_t,parameters)

            #Save Cell and hidden state
            a.append(a_next)
            c.append(c_next)

            #Update loss
            loss = loss + np.sum(- y_t*np.log(y_hat)-(1-y_t)*np.log(1-y_hat))

            #Save Cache
            caches.append(cache_t)


        return a,c,caches,loss

    def LSTM_step_backward(self,da_next,dc_next,cache_t,parameters,gradients):
        
        """
        cahce_t : (y_hat,a_t,c_t,x_t,y_t,a_prev,c_prev,Gamma_o,Gamma_f,Gamma_u,c_st)
        """
        
        y_hat,a_t,c_t,x_t,y_t,a_prev,c_prev,Gamma_o,Gamma_f,Gamma_u,c_st = cache_t

        Wca = parameters["Wca"]
        Wcx = parameters["Wcx"]
        Wua = parameters["Wua"]
        Wux = parameters["Wux"]
        Wfa = parameters["Wfa"]
        Wfx = parameters["Wfx"]
        Woa = parameters["Woa"]
        Wox = parameters["Wox"]
        Wya = parameters["Wya"]


        dZy = y_hat - y_t
        da_t = da_next + np.dot(Wya.T,dZy)
        dc_t = dc_next + da_t*Gamma_o*(1-((np.tanh(c_t))**2))
        dZf = dc_t*c_prev*Gamma_f*(1-Gamma_f)
        dZu = dc_t*c_st*Gamma_u*(1-Gamma_u)
        dZc = dc_t*Gamma_u*(1-(c_st**2))
        dZo = da_t*np.tanh(c_t)*Gamma_o*(1-Gamma_o)

        gradients["dWya"] += np.dot(dZy,a_t.T)
        gradients["dby"] += dZy

        gradients["dWoa"] += np.dot(dZo,a_prev.T)
        gradients["dWox"] += np.dot(dZo,x_t.T)
        gradients["dbo"] += dZo

        gradients["dWca"] += np.dot(dZc,a_prev.T)
        gradients["dWcx"] += np.dot(dZc,x_t.T)
        gradients["dbc"] += dZc

        gradients["dWfa"] += np.dot(dZf,a_prev.T)
        gradients["dWfx"] += np.dot(dZf,x_t.T)
        gradients["dbf"] += dZf

        gradients["dWua"] += np.dot(dZu,a_prev.T)
        gradients["dWux"] += np.dot(dZu,x_t.T)
        gradients["dbu"] += dZu


        da_prev = np.dot(Woa.T,dZo)+np.dot(Wfa.T,dZf)+np.dot(Wua.T,dZu)+np.dot(Wca.T,dZc)
        dc_prev = dc_t*Gamma_f

        return gradients,da_prev,dc_prev


    def LSTM_backward(self,parameters,caches,regularization_factor=0.1):

        """
        cache:
        cache1
        cache2
        .
        .
        .
        
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

        n_a,n_x,n_y = self.n_a,self.n_x,self.n_y
        gradients = {}

        #Set up
        gradients["dWya"] = np.zeros((n_y,n_a))
        gradients["dby"] = np.zeros((n_y,1))

        gradients["dWoa"] = np.zeros((n_a,n_a))
        gradients["dWox"] = np.zeros((n_a,n_x))
        gradients["dbo"] = np.zeros((n_a,1))

        gradients["dWca"] = np.zeros((n_a,n_a))
        gradients["dWcx"] = np.zeros((n_a,n_x))
        gradients["dbc"] = np.zeros((n_a,1))

        gradients["dWfa"] = np.zeros((n_a,n_a))
        gradients["dWfx"] = np.zeros((n_a,n_x))
        gradients["dbf"] = np.zeros((n_a,1))

        gradients["dWua"] = np.zeros((n_a,n_a))
        gradients["dWux"] = np.zeros((n_a,n_x))
        gradients["dbu"] = np.zeros((n_a,1))

        #get shape
        n_a,n_x = parameters["Wux"].shape
        T = len(caches)

        da_next = np.zeros((n_a,1))
        dc_next = np.zeros((n_a,1))

        for t in reversed(range(T)):

            #Get Cache_t
            cache_t = caches[t]

            #Backward 1 step
            gradients,da_next,dc_next = self.LSTM_step_backward(da_next,dc_next,cache_t,parameters,gradients)

        """
        #regularization
        
        gradients["dWya"] = gradients["dWya"]/T + regularization_factor*parameters["Wya"]

        gradients["dWoa"] = gradients["dWoa"]/T + regularization_factor*parameters["Woa"]
        gradients["dWox"] = gradients["dWox"]/T + regularization_factor*parameters["Wox"]

        gradients["dWca"] = gradients["dWca"]/T + regularization_factor*parameters["Wca"]
        gradients["dWcx"] = gradients["dWcx"]/T + regularization_factor*parameters["Wcx"]

        gradients["dWfa"] = gradients["dWfa"]/T + regularization_factor*parameters["Wfa"]
        gradients["dWfx"] = gradients["dWfx"]/T + regularization_factor*parameters["Wfx"]

        gradients["dWua"] = gradients["dWua"]/T + regularization_factor*parameters["Wua"]
        gradients["dWux"] = gradients["dWux"]/T + regularization_factor*parameters["Wux"]
        """

        return gradients


    def optimize(self,X,Y,a0,c0,parameters,v,s,i,regularization_factor=1,beta1=0.9,beta2=0.999,eplison=1e-8,learning_rate=0.001):

        #Get shape
        T_x,n_x = X.shape
        T,n_y = Y.shape

        #Forward Propogation
        a,c,caches,loss = self.LSTM_forward(X,Y,a0,c0,parameters)

        #Backward Propogation
        gradients = self.LSTM_backward(parameters,caches,regularization_factor)

        #gradient clipping
        gradients = self.gradient_clip(gradients,self.maxValue)

        #Update parameters
        parameters,v,s = self.update_parameters_with_Adam(gradients,parameters,v,s,i,beta1,beta2,eplison,learning_rate)
        #parameters = self.update_parameters(gradients,parameters,learning_rate)
        
        return parameters,loss,a[-1],c[-1],v,s

    def model(self,X,Y,iterations = 151,learning_rate=0.001,regularization_factor=0.1,beta1=0.9,beta2=0.999,eplison=1e-8,print_cost=False):

        parameters = self.initialize_parameters(self.n_a,self.n_x,self.n_y)

        a0 = np.random.randn(self.n_a,1)
        c0 = np.random.randn(self.n_a,1)
        T = Y.shape[0]

        loss = 0
        gradients = {}

        #Set up
        n_a,n_y,n_x = self.n_a,self.n_y,self.n_x
        gradients["dWya"] = np.zeros((n_y,n_a))
        gradients["dby"] = np.zeros((n_y,1))

        gradients["dWoa"] = np.zeros((n_a,n_a))
        gradients["dWox"] = np.zeros((n_a,n_x))
        gradients["dbo"] = np.zeros((n_a,1))

        gradients["dWca"] = np.zeros((n_a,n_a))
        gradients["dWcx"] = np.zeros((n_a,n_x))
        gradients["dbc"] = np.zeros((n_a,1))

        gradients["dWfa"] = np.zeros((n_a,n_a))
        gradients["dWfx"] = np.zeros((n_a,n_x))
        gradients["dWfc"] = np.zeros((n_a,n_a))
        gradients["dbf"] = np.zeros((n_a,1))

        gradients["dWua"] = np.zeros((n_a,n_a))
        gradients["dWux"] = np.zeros((n_a,n_x))
        gradients["dWuc"] = np.zeros((n_a,n_a))
        gradients["dbu"] = np.zeros((n_a,1))
        
        v,s = self.initialize_Adam(gradients)

        
        for i in range(iterations):

            parameters,curr_loss,a0,c0,v,s = self.optimize(X,Y,a0,c0,parameters,v,s,i+1,regularization_factor,beta1,beta2,eplison,learning_rate)

            #update loss
            curr_loss = np.sum(curr_loss)
            for para in parameters.values():

                curr_loss = curr_loss #+ regularization_factor*(np.sum(para)**2)/2

            loss = curr_loss


            if print_cost and (i%50) == 0:

                print("Loss :",loss)


        return parameters,a0,c0
        























        
        
