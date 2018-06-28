import numpy as np

#SGD stoped by evaluate metric and loss functions at every epoch. 

class RegressionLineaire():
    def __init__(self, gradient_descent=False, learning_step = 0.01, tolerance = 0.01, stochastic = False, batch_size = 1, epochs = 1000):
        self.coeff = None
        self.gd = gradient_descent
        self.sgd = stochastic
        self.learning_step = learning_step
        self.tolerance = tolerance
        self.batch = batch_size
        self.nb_epochs = epochs
    
    def fit(self, X, Y, X_val = None, Y_val = None):
        p =  X.shape[0]
        X = np.concatenate((np.full((p,1),1), X), axis=1)
        
        
        # Normal equation :
        if not(self.gd):
            self.coeff = np.linalg.multi_dot([(np.linalg.inv(X.T.dot(X))) ,X.T , Y])
            print('coeff lineaire: \n {}'.format(self.coeff))
            return None
        
        # Gradient descent using all lines of X :
        if self.gd and not(self.sgd):
            n = X.shape[1]
            self.coeff = np.random.rand(n,1)
            for _ in range(1000):
                grad = X.T.dot(X.dot(self.coeff) - Y)/ X.shape[0]
                if self.tolerance > np.linalg.norm(grad):
                    return None
                self.coeff = self.coeff - self.learning_step * grad 
            print('coeff gradient descent: \n {}'.format(self.coeff))
        
        # Stockastic descent gradient with random batchs(size 1 by default):
        """if X_val is not None:
            X_val = np.concatenate((np.full((X_val.shape[0],1),1), X_val), axis=1)"""
            
        if self.gd and self.sgd:
            p = X.shape[1]
            n = X.shape[0]
            self.coeff = np.random.rand(p,1)
            
            L =[0]
            M =[np.inf]
            
            for _ in range(self.nb_epochs):
                shuffled_index = np.random.permutation(range(0,X.shape[0]))
                i=0
                size = self.batch
                sum_norm_grad =0
                
                l = []
                m = [] 
                
                for i in range(int(n/size)):
                    lines = shuffled_index[i*size: min(n,(i+1)*size)]
                    X_batch = X[lines,:]
                    Y_batch = Y[lines]
                    grad = X_batch.T.dot(X_batch.dot(self.coeff) - Y_batch)/X_batch.shape[0]
                    self.coeff = self.coeff - self.learning_step * grad
                    
                    #incrementing stops parameters:
                    if X_val is not None and Y_val is not None:
                        loss_batch = (Y-X.dot(self.coeff)).T.dot(Y-X.dot(self.coeff))
                        l.append(loss_batch)
                        m.append(self.evaluate(X_val,Y_val))       
                    sum_norm_grad+= np.linalg.norm(grad)
                    
                if X_val is not None and Y_val is not None:
                    L.append(np.mean(l))
                    M.append(np.mean(m))
                    
                    if M[-2]< M[-1]:
                        print("Metric : {}".format(M))
                        print("Loss : {}".format(L))
                        return None
                
                if self.tolerance > sum_norm_grad:
                    return None
                
        
            print('coeff stochastic gradient descent:  \n {}'.format(self.coeff))           
                
        
    
    def predict(self, A):
        p =  A.shape[0]
        X = np.concatenate((np.full((p,1),1), A), axis=1)
        return X.dot(self.coeff)
    
    def evaluate(self,X,Y):
        E = (Y - self.predict(X))
        return E.var()

