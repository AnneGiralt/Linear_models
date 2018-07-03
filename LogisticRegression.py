
import numpy as np

#SGD stoped by evaluate metric and loss functions at every epoch. 

class LogisticRegression():
    def __init__(self, learning_step = 0.1, tolerance = 0.01, stochastic = False, batch_size = 50, epochs = 1000):
        self.coeff = None
        self.sgd = stochastic
        self.learning_step = learning_step
        self.tolerance = tolerance
        self.batch = batch_size
        self.nb_epochs = epochs
    
    def fit(self, X, Y, X_val = None, Y_val = None):
        p =  X.shape[0]
        X = np.concatenate((np.full((p,1),1), X), axis=1)
        
        
        # Gradient descent using all lines of X :
        if not(self.sgd):
            n = X.shape[1]
            self.coeff = np.random.rand(n,1)
            for _ in range(self.nb_epochs):
                grad = X.T.dot(sig(X.dot(self.coeff)) - Y)/X.shape[0]
                
                if self.tolerance > np.linalg.norm(grad):
                    print('coeff gradient descent (tolerence): \n {}'.format(self.coeff))
                    return None
                self.coeff = self.coeff - self.learning_step * grad 
            print('coeff gradient descent(epoch): \n {}'.format(self.coeff))
        
        # Stockastic descent gradient with random batchs(size 30 by default):            
        if self.sgd:
            p = X.shape[1]
            n = X.shape[0]
            self.coeff = np.ones((p,1))
            
            L =[]
            M =[np.inf, np.inf]
            
            
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
                    
                    grad = X_batch.T.dot(sig(X_batch.dot(self.coeff)) - Y_batch)/X_batch.shape[0]
                    self.coeff = self.coeff - self.learning_step * grad
                    

                    #incrementing stops parameters:
                    if X_val is not None and Y_val is not None:
                        l.append(self.loss(X_batch,Y_batch))
                        m.append(self.evaluate(X_val,Y_val))  
                    sum_norm_grad+= np.linalg.norm(grad)
                    
                    L.append(self.loss(X,Y))
                
                    
                if X_val is not None and Y_val is not None:
                    L.append(np.mean(l))
                    M.append(np.mean(m))
                    
                    if M[-1]> M[-2] and M[-2] >= M[-3]:
                        print('coeff stochastic gradient descent (overfit):  \n {}'.format(self.coeff))
                        print("Metric : {}".format(M))
                        print("Loss : {}".format(L))
                        return None
                
                if self.tolerance > sum_norm_grad:
                    print('coeff stochastic gradient descent (tolerence):  \n {}'.format(self.coeff))
                    return None 
        
            print('coeff stochastic gradient descent (epoch):  \n {}'.format(self.coeff))           
                
        
    
    def predict(self, A):
        p =  A.shape[0]
        X = np.concatenate((np.full((p,1),1), A), axis=1)
        return sig(X.dot(self.coeff))
    
    #Error/nb_prediction
    def evaluate(self,X,Y):
        E = Y - np.around(self.predict(X))   
        return np.sum(np.absolute(E))/E.shape[0]
    
    def loss(self,X,Y):
        v = Y*np.log(X.dot(sig(self.coeff)))+ (np.ones((Y.shape[0],1))-Y)*np.log(np.ones((X.shape[0],1))-sig(X.dot(self.coeff)))
        return - np.sum(v)/X.shape[0]    


def sig(x):
    return 1/(1+ np.exp(-x))  