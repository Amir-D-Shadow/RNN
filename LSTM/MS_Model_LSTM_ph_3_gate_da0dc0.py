import numpy as np
import os

class MS_Model_LSTM:

    def __init__(self,n_a=128,n_x=49,n_y=49,max_val=0.8,iterations = 201,learning_rate=0.055,regularization_factor=1,beta1=0.9,beta2=0.999,eplison=1e-8):

        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.maxValue = max_val

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization_factor = regularization_factor
        self.beta1 = beta1
        self.beta2 = beta2
        self.eplison = eplison

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

        return np.where(z>=0,(1/(1+np.exp(-z))),(np.exp(z)/(1 + np.exp(z))))

    def relu(self,z):

        return np.maximum(z,0)

    def activation_derivative(self,z,activation="tanh"):

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

    def initialize_Adam(self,parameters):

        #Set up
        v = {}
        s = {}

        for para in parameters.keys():

            grad = "d" + para

            v[grad] = np.zeros_like(parameters[para])
            s[grad] = np.zeros_like(parameters[para])

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
        
            

    def gradient_clip(self,gradients,maxValue):

        for grad in gradients.values():

            np.clip(a=grad,a_min=-maxValue,a_max=maxValue,out=grad)

        return gradients
    

    def initialize_parameters(self,n_a=128,n_x=49,n_y=49):

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
        Woc : (n_a,n_a)
        bo : (n_a,1)

        y_hat:

        Wya: (n_y,n_a)
        by: (n_y,1)
        
        """

        parameters = {}

        parameters["a0"] = np.random.randn(n_a,1)*np.sqrt(2/n_a)
        parameters["c0"] = np.random.randn(n_a,1)*np.sqrt(2/n_a)

        parameters["Wca"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["Wcx"] = np.random.randn(n_a,n_x)*np.sqrt(2/(n_a+n_x))
        parameters["bc"] = np.zeros((n_a,1))

        parameters["Wua"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["Wux"] = np.random.randn(n_a,n_x)*np.sqrt(2/(n_a+n_x))
        parameters["Wuc"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["bu"] = np.zeros((n_a,1))

        parameters["Wfa"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["Wfx"] = np.random.randn(n_a,n_x)*np.sqrt(2/(n_a+n_x))
        parameters["Wfc"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["bf"] = np.zeros((n_a,1))

        parameters["Woa"] = np.random.randn(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["Wox"] = np.random.randn(n_a,n_x)*np.sqrt(2/(n_a+n_x))
        parameters["Woc"] = np.random.rand(n_a,n_a)*np.sqrt(2/(n_a+n_a))
        parameters["bo"] = np.zeros((n_a,1))

        parameters["Wya"] = np.random.randn(n_y,n_a)*np.sqrt(2/(n_a+n_y))
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
        Wuc = parameters["Wuc"]
        bu = parameters["bu"]

        Wfa = parameters["Wfa"]
        Wfx = parameters["Wfx"]
        Wfc = parameters["Wfc"]
        bf = parameters["bf"]

        Woa = parameters["Woa"]
        Wox = parameters["Wox"]
        Woc = parameters["Woc"]
        bo = parameters["bo"]

        Wya = parameters["Wya"]
        by = parameters["by"]


        #~C_t
        z = np.dot(Wca,a_prev)+np.dot(Wcx,x_t)+bc
        c_st = np.tanh(z)

        #Update Gate
        z = np.dot(Wua,a_prev)+np.dot(Wux,x_t)+np.dot(Wuc,c_prev)+bu
        #print("Update Gate: ",np.sum(z))
        Gamma_u = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Forget Gate
        z = np.dot(Wfa,a_prev)+np.dot(Wfx,x_t)+np.dot(Wfc,c_prev)+bf
        #print("Forget Gate: ",np.sum(z))
        Gamma_f = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Output Gate
        z = np.dot(Woa,a_prev)+np.dot(Wox,x_t)+np.dot(Woc,c_prev)+bo
        #print("Output Gate: ",np.sum(z))
        Gamma_o = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Cells state t
        c_t = Gamma_u*c_st+Gamma_f*c_prev

        #Hidden state t
        a_t = Gamma_o*np.tanh(c_t)

        #y_hat prediction
        z = np.dot(Wya,a_t)+by
        #print("y_hat: ",np.sum(z))
        y_hat = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

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
            loss = loss + np.sum(- y_t*np.log(y_hat+(1e-12))-(1-y_t)*np.log(1-y_hat+(1e-12)))

            #Save Cache
            caches.append(cache_t)


        return a,c,caches,loss

    def LSTM_step_backward(self,da_next,dc_next,cache_t,parameters,gradients):
        
        """
        cahce_t : (y_hat,a_t,c_t,x_t,y_t,a_prev,c_prev,Gamma_o,Gamma_f,Gamma_u,c_st)
        """
        
        y_hat,a_t,c_t,x_t,y_t,a_prev,c_prev,Gamma_o,Gamma_f,Gamma_u,c_st = cache_t

        Woa = parameters["Woa"]
        Woc = parameters["Woc"]
        Wfa = parameters["Wfa"]
        Wfc = parameters["Wfc"]
        Wua = parameters["Wua"]
        Wuc = parameters["Wuc"]
        Wca = parameters["Wca"]
        Wya = parameters["Wya"]

        dZy = y_hat - y_t
        da_t = da_next + np.dot(Wya.T,dZy)
        
        dc_t = dc_next + da_t*Gamma_o*(1-((np.tanh(c_t))**2))
        #dc_t = np.where(dc_t>=0,np.minimum(dc_t,5e3),np.maximum(dc_t,-5e3))

        dZf = dc_t*c_prev*Gamma_f*(1-Gamma_f)
        dZu = dc_t*c_st*Gamma_u*(1-Gamma_u)
        dZc = dc_t*Gamma_u*(1-((c_st)**2))
        dZo = da_t*np.tanh(c_t)*Gamma_o*(1-Gamma_o)

        gradients["dWya"] += np.dot(dZy,a_t.T)
        gradients["dby"] += dZy

        gradients["dWoa"] += np.dot(dZo,a_prev.T)
        gradients["dWox"] += np.dot(dZo,x_t.T)
        gradients["dWoc"] += np.dot(dZo,c_prev.T)
        gradients["dbo"] += dZo

        gradients["dWca"] += np.dot(dZc,a_prev.T)
        gradients["dWcx"] += np.dot(dZc,x_t.T)
        gradients["dbc"] += dZc

        gradients["dWfa"] += np.dot(dZf,a_prev.T)
        gradients["dWfx"] += np.dot(dZf,x_t.T)
        gradients["dWfc"] += np.dot(dZf,c_prev.T)
        gradients["dbf"] += dZf

        gradients["dWua"] += np.dot(dZu,a_prev.T)
        gradients["dWux"] += np.dot(dZu,x_t.T)
        gradients["dWuc"] += np.dot(dZu,c_prev.T)
        gradients["dbu"] += dZu


        da_prev = np.dot(Woa.T,dZo)+np.dot(Wfa.T,dZf)+np.dot(Wua.T,dZu)+np.dot(Wca.T,dZc)
        dc_prev = dc_t*Gamma_f+np.dot(Wfc.T,dZf)+np.dot(Wuc.T,dZu)+np.dot(Woc.T,dZo)

        return gradients,da_prev,dc_prev


    def LSTM_backward(self,parameters,cache,regularization_factor=0.1):

        """
        cache:
        cache1
        cache2
        .
        .
        .
        
        """

        #initialize gradient
        gradients = {}
        
        for para in parameters.keys():

            grad = "d" + para
            gradients[grad] = np.zeros(parameters[para].shape)

                  
        n_a,n_x = parameters["Wux"].shape
        T = len(cache)

        da_next = np.zeros((n_a,1))
        dc_next = np.zeros((n_a,1))

        for t in reversed(range(T)):

            #Get Cache_t
            cache_t = cache[t]

            #Backward 1 step
            gradients,da_next,dc_next = self.LSTM_step_backward(da_next,dc_next,cache_t,parameters,gradients)


        #gradient for a0 and c0
        gradients["da0"] = da_next
        gradients["dc0"] = dc_next
        
        #regularization
        
        gradients["dWya"] = (gradients["dWya"]+regularization_factor*parameters["Wya"])/T

        gradients["dWoa"] = (gradients["dWoa"]+regularization_factor*parameters["Woa"])/T
        gradients["dWox"] = (gradients["dWox"]+regularization_factor*parameters["Wox"])/T
        gradients["dWoc"] = (gradients["dWoc"]+regularization_factor*parameters["Woc"])/T

        gradients["dWca"] = (gradients["dWca"]+regularization_factor*parameters["Wca"])/T
        gradients["dWcx"] = (gradients["dWcx"]+regularization_factor*parameters["Wcx"])/T

        gradients["dWfa"] = (gradients["dWfa"]+regularization_factor*parameters["Wfa"])/T
        gradients["dWfx"] = (gradients["dWfx"]+regularization_factor*parameters["Wfx"])/T
        gradients["dWfc"] = (gradients["dWfc"]+regularization_factor*parameters["Wfc"])/T

        gradients["dWua"] = (gradients["dWua"]+regularization_factor*parameters["Wua"])/T
        gradients["dWux"] = (gradients["dWux"]+regularization_factor*parameters["Wux"])/T
        gradients["dWuc"] = (gradients["dWuc"]+regularization_factor*parameters["Wuc"])/T


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

        
        return parameters,loss,a[-1],c[-1],parameters["a0"],parameters["c0"],v,s



    def model(self,X,Y,iterations = 51,loss_threshold=20,learning_rate=0.001,regularization_factor=0.1,beta1=0.9,beta2=0.999,eplison=1e-8,print_cost=False):

        #a0 = np.random.randn(self.n_a,1)
        #c0 = np.random.randn(self.n_a,1)
        T = Y.shape[0]

        loss = 0
        gradients = {}

        #Set up
        parameters = self.initialize_parameters(self.n_a,self.n_x,self.n_y)
        v,s = self.initialize_Adam(parameters)
        a0 = parameters["a0"]
        c0 = parameters["c0"]

        #Train the model
        for i in range(iterations):

            parameters,curr_loss,a_T,c_T,a0,c0,v,s = self.optimize(X,Y,a0,c0,parameters,v,s,i+1,regularization_factor,beta1,beta2,eplison,learning_rate)

            #update loss
            curr_loss = np.sum(curr_loss)

            for para in parameters.keys():

                curr_loss += regularization_factor*(np.sum(parameters[para]**2))/2
            

            loss = curr_loss/T


            if print_cost and (i%50) == 0:

                print("Loss :",loss)

            #call back
            if loss < loss_threshold:

                print("Loss :",loss)

                break

                
        #Save the model
        path = os.getcwd() + "/model/"

        for para in parameters.keys():

            np.savetxt(path+para,parameters[para],delimiter=",")

        return parameters,a_T,c_T


    def predict(self,a_prev,c_prev,x_t,parameters):
        

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
        Woc = parameters["Woc"]
        bo = parameters["bo"]

        Wya = parameters["Wya"]
        by = parameters["by"]


        #~C_t
        z = np.dot(Wca,a_prev)+np.dot(Wcx,x_t)+bc
        c_st = np.tanh(z)

        #Update Gate
        z = np.dot(Wua,a_prev)+np.dot(Wux,x_t)+np.dot(Wuc,c_prev)+bu
        #print("Update Gate: ",np.sum(z))
        Gamma_u = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Forget Gate
        z = np.dot(Wfa,a_prev)+np.dot(Wfx,x_t)+np.dot(Wfc,c_prev)+bf
        #print("Forget Gate: ",np.sum(z))
        Gamma_f = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Output Gate
        z = np.dot(Woa,a_prev)+np.dot(Wox,x_t)+np.dot(Woc,c_prev)+bo
        #print("Output Gate: ",np.sum(z))
        Gamma_o = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

        #Cells state t
        c_t = Gamma_u*c_st+Gamma_f*c_prev

        #Hidden state t
        a_t = Gamma_o*np.tanh(c_t)

        #y_hat prediction
        z = np.dot(Wya,a_t)+by
        #print("y_hat: ",np.sum(z))
        y_hat = self.sigmoid(np.where(z>=0,np.minimum(z,1e2),np.maximum(z,-1e2)))

    
        res = y_hat.flatten().argsort()[-7:]

        return res,a_t,c_t
        























        
        
