##########################################################
#          MULTILAYER PERCEPTRON FROM SCRATCH            #
##########################################################

# import packages
import numpy as np
import math

# class for Multilayer Perceptron
class MLPAlgorithm(object):
    
    # hyperparameters definition
    def __init__(self, eta, threshold, max_epochs):
        self.eta = eta
        self.threshold = threshold
        self.max_epochs = max_epochs
    
    # function to define MLP architecture
    def build_architecture(self, input_length, hidden_length, output_length):
        # hyperparameters for MLP architecture
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        
        # random initialization of weights and biases
        # hidden layer [b x (a+1)]
        self.Wh = np.random.rand(self.hidden_length, self.input_length)
        self.bh = np.random.rand(self.hidden_length)        
        # output layer [c x (b+1)]
        self.Wo = np.random.rand(self.output_length, self.hidden_length)
        self.bo = np.random.rand(self.output_length)
        
        return self

    # linear combination    
    def input_net(self, x, w, b):
        net = np.dot(w, x) + b
        return net
    
    # sigmoid activation function
    def f(self, net):
        return 1/(1 + math.exp(-net))
    
    # derivate of sigmoid activation function
    def df(self, fnet):
        return fnet * (1 - fnet)
    
    # forward step
    def forward(self, x):    
        # hidden layer
        self.net_h = self.input_net(x, self.Wh, self.bh)
        self.fnet_h = np.array([self.f(net) for net in self.net_h])
        self.dfnet_h = np.array([self.df(fnet) for fnet in self.fnet_h])
        
        # output layer
        self.net_o = self.input_net(self.fnet_h, self.Wo, self.bo)
        self.fnet_o = np.array([self.f(net) for net in self.net_o])
        self.dfnet_o = np.array([self.df(fnet) for fnet in self.fnet_o])
    
    # function to make prediction given a point
    def predict(self, xi):
        self.forward(xi)
        logits = self.fnet_o

        if self.output_length == 1:
            pred = np.where(self.fnet_o >= 0.5, 1, 0)[0]
        else:
            pred = np.argmax(self.fnet_o)[0]
        
        return logits, pred
    
    # iterative training step
    def fit(self, x_train, y_train):
        n = x_train.shape[0]
        E = 2 * self.threshold
        count = 0
        cost = list()
        
        # training in each epoch
        while(E >= self.threshold and count <= self.max_epochs + 1):
            E = 0
            
            # stochastic gradient descendent algorithm (SGD) -> for each sample
            for i in range(n):
                xi = x_train[i, :]
                
                if self.output_length > 1:  # for multi-classification problems
                    yi = y_train[i, :]
                else:                        # for binary-classification problems
                    yi = y_train[i]
                
                self.forward(xi)
                y_pred = self.fnet_o
                
                # calculate error
                error = (yi - y_pred)
                E = E + sum(error**2)
                
                #Output Layer
                #------------#
                
                # initialize gradient weights Wo for output layer
                dE_dWo = np.zeros(
                        self.output_length * self.hidden_length).reshape(
                                self.output_length, self.hidden_length)
                
                # initialize gradient bias bo for output layer       
                dE_dbo = np.zeros(self.output_length)
                                
                # calculate delta for output layer
                delta_o = -error * self.dfnet_o                                                 
                
                # iterate for each neuron in output layer                                             
                for j in range(self.output_length):
                    # iterate in each sinapsis related with j-th output neuron                    
                    for i in range(self.hidden_length):    
                        # dE_dWo
                        dE_dWo[j, i] = delta_o[j] * self.fnet_h[i] 
                        self.Wo[j, i] = self.Wo[j, i] - self.eta * dE_dWo[j, i]
                    # dE_dbo
                    dE_dbo[j] = delta_o[j] * 1
                    self.bo[j] = self.bo[j] - self.eta * dE_dbo[j]
                                                                        
                #Hidden Layer
                #------------#
                
                # initialize gradient weights Wh for hidden layers
                dE_dWh = np.zeros(
                        self.hidden_length * self.input_length).reshape(
                                self.hidden_length, self.input_length)
                
                # initialize gradient biases bh for hidden layers
                dE_dbh = np.zeros(self.hidden_length)
                
                # calculate delta for hidden layer                
                delta_h = self.dfnet_h * np.dot(delta_o, self.Wo)
                
                # iterate for each neuron of hidden layers
                for i in range(self.hidden_length):
                    # iterate in each sinapsis related with i-th hidden neuron                                
                    for k in range(self.input_length):
                        #dE_dWh
                        dE_dWh[i, k] = delta_h[i] * xi[k]
                        self.Wh[i, k] = self.Wh[i, k] - self.eta * dE_dWh[i, k]
                    #dE_dbh
                    dE_dbh[i] = delta_h[i] * 1
                    self.bh[i] = self.bh[i] - self.eta * dE_dbh[i]
            
            # count number of epochs
            count = count + 1
            
            # calculate mean square error (MSE)     
            E = round(1/2 * (E/n), 5)
            cost.append(E)
            
            # report results each 100 epochs
            if(count%100 == 0):
                print('Epoch ', count, ': loss = ', E)
                            
        # store results in MLP class
        self.epochs = count
        self.loss_ = E
        self.cost_ = cost
        
        return self

    # function to make iterative process of test    
    def test(self, x_test, y_test):
        n = x_test.shape[0]
        self.accuracy = 0
        y_prob = list()
        y_pred = list()
        
        for i in range(n):
            xi = x_test[i, :]
            
            if self.output_length > 1:  # for multi-classification problems
                yi = y_test[i, :]
            else:                       # for binary classification problems
                yi = y_test[i]
            
            # given inputs, predict class and probabilities
            logits, pred = self.predict(xi)
            y_prob.append(logits)
            y_pred.append(pred)
            
            # verify correct classifications
            if np.array_equal(y_pred[i], yi):
                self.accuracy += 1
            
        # calculate accuracy
        self.accuracy = 100 * round(self.accuracy/n, 5)

        return y_pred
    