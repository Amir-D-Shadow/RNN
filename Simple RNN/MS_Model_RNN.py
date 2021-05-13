import numpy as np
import os


class MS_Model_RNN:

    def __init__(self,n_a=128,n_x=49,n_y=49,max_val=5):

        self.Train_Data_X = None
        self.Train_Data_Y = None
        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y

        self.a0 = np.random.randn(n_a,1)
        self.maxValue = max_val


    def load_data(self):

        Path = os.getcwd() + "/" + "Data" + "/" + "MarkSixData.txt"
        data_x = np.loadtxt(Path,delimiter=",",dtype=int)
        data_list = data_x.tolist()

        self.Train_Data_X = np.zeros((data_x.shape[0],49))

        row_index = 0

        for row in data_list:

            for col in row:

                self.Train_Data_X[row_index][col-1] = 1

            row_index += 1

        self.Train_Data_Y = self.Train_Data_X[:-1,:].copy()
        

    def initial_parameters(self,n_a,n_x,n_y):

        """
        Parameters Contain:
        
         Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
         Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
         Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
         ba --  Bias, numpy array of shape (n_a, 1)
         by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        """
        Wax = np.random.randn(self.n_a,self.n_x)*0.01
        Waa = np.random.randn(self.n_a,self.n_a)*0.01
        Wya = np.random.randn(self.n_y,self.n_a)*0.01
        ba = np.random.randn(self.n_a,1)
        by = np.random.randn(self.n_y,1)
        
        return { "Wax":Wax,"Waa":Waa,"Wya":Wya,"ba":ba,"by":by}

        

    def sigmoid(self,z):

        cache = z

        return np.where(z>=0,(1/(1+np.exp(-z))),(np.exp(z)/(1 + np.exp(z)))) ,cache

    def relu(self,z):

        cache = z

        return np.maximum(0,z) , cache

    def relu_derivative(self,z):

        drelu = np.full(shape=z.shape,fill_value= 1)
        
        drelu[z<0] = 0

        return drelu

    def initialize_Adam(self,parameters):

        v = {}
        s = {}

        Waa = parameters["Waa"]
        Wax = parameters["Wax"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        v["dWaa"] = np.zeros_like(Waa)
        v["dWax"] = np.zeros_like(Wax)
        v["dWya"] = np.zeros_like(Wya)
        v["dba"] = np.zeros_like(ba)
        v["dby"] = np.zeros_like(by)

        s["dWaa"] = np.zeros_like(Waa)
        s["dWax"] = np.zeros_like(Wax)
        s["dWya"] = np.zeros_like(Wya)
        s["dba"] = np.zeros_like(ba)
        s["dby"] = np.zeros_like(by)

        return v,s

    def update_parameters_with_Adam(self,gradients,parameters,v,s,t,learning_rate,beta1,beta2,eplison):

        """
        Adam update parameters per ITERATIONS!!!

        """

        #Important: v,s and v_corrected,s_corrected should be treated separately
        v_corrected = {}
        s_corrected = {}

        dWya = gradients["dWya"]
        dWaa = gradients["dWaa"]
        dWax = gradients["dWax"]
        dba =  gradients["dba"]
        dby = gradients["dby"]


        #Update v
        v["dWaa"] = beta1*v["dWaa"] + (1-beta1)*gradients["dWaa"]
        v["dWax"] = beta1*v["dWax"] + (1-beta1)*gradients["dWax"]
        v["dWya"] = beta1*v["dWya"] + (1-beta1)*gradients["dWya"]
        v["dba"] = beta1*v["dba"] + (1-beta1)*gradients["dba"]
        v["dby"] = beta1*v["dby"] + (1-beta1)*gradients["dby"]

        v_corrected["dWaa"] = v["dWaa"]/(1-beta1**t)
        v_corrected["dWax"] = v["dWax"]/(1-beta1**t)
        v_corrected["dWya"] = v["dWya"]/(1-beta1**t)
        v_corrected["dba"] = v["dba"]/(1-beta1**t)
        v_corrected["dby"] = v["dby"]/(1-beta1**t)

        #update s
        s["dWaa"] = beta2*s["dWaa"] + (1-beta2)*(gradients["dWaa"]**2)
        s["dWax"] = beta2*s["dWax"] + (1-beta2)*(gradients["dWax"]**2)
        s["dWya"] = beta2*s["dWya"] + (1-beta2)*(gradients["dWya"]**2)
        s["dba"] = beta2*s["dba"] + (1-beta2)*(gradients["dba"]**2)
        s["dby"] = beta2*s["dby"] + (1-beta2)*(gradients["dby"]**2)

        s_corrected["dWaa"] = s["dWaa"]/(1-beta2**t)
        s_corrected["dWax"] = s["dWax"]/(1-beta2**t)
        s_corrected["dWya"] = s["dWya"]/(1-beta2**t)
        s_corrected["dba"] = s["dba"]/(1-beta2**t)
        s_corrected["dby"] = s["dby"]/(1-beta2**t)

        #Update parameters
        parameters["Waa"] -= learning_rate*(v_corrected["dWaa"])/(np.sqrt(s_corrected["dWaa"]+eplison))
        parameters["Wax"] -= learning_rate*(v_corrected["dWax"])/(np.sqrt(s_corrected["dWax"]+eplison))
        parameters["Wya"] -= learning_rate*(v_corrected["dWya"])/(np.sqrt(s_corrected["dWya"]+eplison))
        parameters["ba"] -= learning_rate*(v_corrected["dba"])/(np.sqrt(s_corrected["dba"]+eplison))
        parameters["by"] -= learning_rate*(v_corrected["dby"])/(np.sqrt(s_corrected["dby"]+eplison))


        return parameters,v,s                
        

        

    def gradient_clip(self,gradients,max_val=5):


        dWya = gradients["dWya"]
        dWaa = gradients["dWaa"]
        dWax = gradients["dWax"]
        dba =  gradients["dba"]
        dby = gradients["dby"]
   
        for gradient in gradients.values():
            
            np.clip(a=gradient,a_min=-max_val,a_max=max_val,out=gradient)
    
        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
    
        return gradients


    def update_parameters(self,gradients,parameters,learning_rate = 0.01):

        dWya = gradients["dWya"]
        dWaa = gradients["dWaa"]
        dWax = gradients["dWax"]
        dba =  gradients["dba"]
        dby = gradients["dby"]

        parameters["Waa"] -= learning_rate*dWaa
        parameters["Wax"] -= learning_rate*dWax
        parameters["Wya"] -= learning_rate*dWya
        parameters["ba"] -= learning_rate*dba
        parameters["by"] -= learning_rate*dby

        return parameters

        
    
    def rnn_forward_1_step(self,x_t,a_prev,parameters):

        """
        x -- (n_x,1)
        a_prev -- (n_a,1)

        Parameters :
        
        Waa -- (n_a, n_a)
        Wax -- (n_a, n_x)
        Wya -- (n_y, n_a)
        ba -- (n_a, 1)
        by -- (n_y, 1)

        a_next -- (n_a,1)
        y_hat -- (n_y,1)

        """
        n_x = x_t.shape[0]
        
        Waa = parameters["Waa"]
        Wax = parameters["Wax"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        x = x_t.reshape(n_x,1).copy()
        a_next,_ = self.relu(np.dot(Waa,a_prev)+np.dot(Wax,x)+ba)
        y_hat,_ = self.sigmoid(np.dot(Wya,a_next)+by)

        cache_1_step = (y_hat,a_next,a_prev,x_t)

        return a_next,y_hat,cache_1_step


    def rnn_forward(self,X,Y,a0,parameters):

        """
        cache a y_pred:
        
        a -- (n_a,T_x)
        y_pred -- (n_y,T_x)

        Data :

        X -- (T_x,n_x)
        Y -- (T_x - 1, n_x)

        y_hat -- (n_y,1)
        y_real -- (n_y,1)

        cache_t:
        y_hat,a_next,a_prev,x_t,parameters

        """

        a_next = a0.copy()
        
        T_x, n_x = X.shape
        n_y,n_a = parameters["Wya"].shape

        y_pred = []
        a = []

        #initialize loss
        y_row = T_x - 1
        loss = 0

        #Current Time Step
        t = 0

        caches = []

        for i in reversed(range(T_x-1)):

            #Get One step data : x_t -- (1,n_x)
            x_t = X[i+1,:]

            #Forward One Step
            a_next,y_hat,cache_t = self.rnn_forward_1_step(x_t,a_next,parameters)

            #Save hidden state a
            a.append(a_next)

            #Save y_hat
            y_pred.append(y_hat)

            #Update loss 
            y_real = Y[i,:].reshape(n_y,1).copy()
            loss += (-y_real*np.log(y_hat)-(1-y_real)*np.log(1-y_hat))

            #Update Time Step
            t += 1

            #save cache
            caches.append(cache_t)

        loss = np.sum(loss)

        return a,y_pred,caches,loss

    def rnn_backward_1_step(self,da_prevt,gradients,parameters,cache_t,y_t):

        """
        gradients :

        dWya -- (n_y,n_a)
        dWaa -- (n_a,n_a)
        dWax -- (n_a,n_x)
        dba -- (n_a,1)
        dby -- (n_y,1)

        parameters:
        Waa -- (n_a, n_a)
        Wax -- (n_a, n_x)
        Wya -- (n_y, n_a)
        ba -- (n_a, 1)
        by -- (n_y, 1)

        cache_t:
        y_hat_t,a_next,a_prev,x_t
        
        """

        y_hat_t,a_t,a_prevt,x_t = cache_t

        x_t = x_t.reshape(self.n_x,1)

        Wya = parameters["Wya"]
        Waa = parameters["Waa"]
        Wax = parameters["Wax"]
        ba = parameters["ba"]
        by = parameters["by"]

        z = np.dot(Waa,a_prevt) + np.dot(Wax,x_t) + ba

        day_t = np.dot(Wya.T,(y_hat_t-y_t))
        da_t = da_prevt + day_t

        gradients["dWya"] += np.dot((y_hat_t-y_t),a_t.T)
        gradients["dby"] += y_hat_t-y_t
        gradients["dWax"] += np.dot((da_t*self.relu_derivative(z)),x_t.T)
        gradients["dWaa"] += np.dot((da_t*self.relu_derivative(z)),a_prevt.T)
        gradients["dba"] += da_t*self.relu_derivative(z)

        da_prevt = np.dot(Waa.T,(da_t*self.relu_derivative(z)))

        
        
        return da_prevt,gradients

    def rnn_backward(self,caches,Y,parameters):

        """
        Y -- (T_x-1,n_y)
        
        cache :
        
        cache 1
        cache 2
        cache 3
        .
        .
        .

        """

        Ty,n_y = Y.shape

        Wya = parameters["Wya"]
        Waa = parameters["Waa"]
        Wax = parameters["Wax"]
        ba = parameters["ba"]
        by = parameters["by"]

        #Initialize Gradient
        dWya = np.zeros_like(Wya)
        dWaa = np.zeros_like(Waa)
        dWax = np.zeros_like(Wax)
        dba = np.zeros_like(ba)
        dby = np.zeros_like(by)

        da_prevt = np.zeros((self.n_a,1))
        
        gradients = {"dWya":dWya,"dWaa":dWaa,"dWax":dWax,"dba":dba,"dby":dby}

        #Time Step
        t = 0

        for i in range(Ty):

            #Retrieve cache
            y_t = Y[i,:].copy().reshape(n_y,1)
            cache_t = caches[t]

            #back propagate once
            da_prevt,gradients = self.rnn_backward_1_step(da_prevt,gradients,parameters,cache_t,y_t)        
            
            
        return gradients


    def optimize(self,parameters,a0,X,Y,v,s,t,learning_rate,beta1,beta2,eplison):

        #forward propagation
        a,y_pred,caches,loss = self.rnn_forward(X,Y,a0,parameters)

        #backward propagation
        gradients = self.rnn_backward(caches,Y,parameters)

        #gradients clip
        gradients = self.gradient_clip(gradients,max_val=self.maxValue)

        
        #update parameter
        """
        parameters = self.update_parameters(gradients,parameters,learning_rate)
        """
        parametres,v,s =self.update_parameters_with_Adam(gradients,parameters,v,s,t,learning_rate,beta1,beta2,eplison)
        
        return parameters,loss,a[-1],v,s


    def model(self,X,Y,num_iterations = 250,learning_rate=0.001,beta1=0.9,beta2=0.999,eplison=1e-8,print_cost= False):

        parameters = self.initial_parameters(self.n_a,self.n_x,self.n_y)

        a_prev = np.zeros((self.n_a,1))

        loss = 0

        v,s = self.initialize_Adam(parameters)

        for i in range(num_iterations):

            parameters,curr_loss,a_prev,v,s = self.optimize(parameters,a_prev,X,Y,v,s,i+1,learning_rate,beta1,beta2,eplison)

            loss = (loss*0.999 + curr_loss*0.001)/Y.shape[0]

            if print_cost and (i%50 == 0):

                print("Curr_loss = ",loss)

        return parameters,a_prev
