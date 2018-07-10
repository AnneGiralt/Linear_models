import numpy as np


#We suppose that every class of Y is given as a vector of dimension the number of classes which is composed uniquely with a unique 1 and zeros.

class SoftmaxRegression():
    def __init__(self, learning_step = 0.01, tolerance = 0.01, stochastic = False, batch_size = 50, epochs = 1000):
        self.coeff = None
        self.sgd = stochastic
        self.learning_step = learning_step
        self.tolerance = tolerance
        self.batch = batch_size
        self.nb_epochs = epochs
        self.k =0
    
    def fit(self, X, Y, X_val = None, Y_val = None):
        self.k =Y[0].shape[0]
        n = X.shape[0]
        X = np.concatenate((np.full((n,1),1), X), axis=1)
        p =  X.shape[1]
        
        # Number of different classes
        k = Y[0].shape[0]
        
        
        # Gradient descent using all lines of X :
        if not(self.sgd):
            
            self.coeff = np.random.rand(p,k)
            
            for _ in range(self.nb_epochs):
                G = np.zeros((p,k)) 
                
                # Constructing the a-th column of G
                for a in range(k):
                    
                    # Initialize p_i^a column
                    P = np.zeros((n,1))
                    for i in range(n):
                        P[i]= proba(a, X[i,:], self.coeff)
                    
                    G[:,a:a+1] = (X.T.dot( P - Y[:,a:a+1]))/X.shape[0]
                
                
                if self.tolerance > np.linalg.norm(G):
                    print('coeff gradient descent (tolerence): \n {}'.format(self.coeff))
                    return None
                self.coeff = self.coeff - self.learning_step * G
                
            print('coeff gradient descent(epoch): \n {}'.format(self.coeff))
        
        # Stockastic descent gradient with random batchs(size 30 by default):            
        if self.sgd:
            p = X.shape[1]
            n = X.shape[0]
            self.coeff = np.zeros((p,k))
            
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
                    n_b = X_batch.shape[0]
                
                    G = np.zeros((p,k))
                    
                    for a in range(k):
            
                        # Initialize p_i^a column
                        P = np.zeros((n_b,1))
                        for i in range(n_b):
                            P[i]= proba(a, X_batch[i,:], self.coeff)
                    
                        G[:,a:a+1] = (X_batch.T.dot(P - Y_batch[:,a:a+1]))/X_batch.shape[0]
                    
                    self.coeff = self.coeff - self.learning_step * G
                    
                    

                    #incrementing stops parameters:
                    if X_val is not None and Y_val is not None:
                        l.append(self.loss(X_batch,Y_batch))
                        m.append(self.evaluate(X_val,Y_val))  
                    sum_norm_grad+= np.linalg.norm(G)
                    
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
        n =  A.shape[0]
        X = np.concatenate((np.full((n,1),1), A), axis=1)
        
        P = np.zeros((n,self.k))
        
        for i in range(n):
            for j in range(self.k):
                P[i,j]= proba(j, X[i,:], self.coeff)
        return P
    
    #Error/nb_prediction
    def evaluate(self,X,Y):
        n = X.shape[0]
        E = np.argmax(Y, axis=1) - np.argmax(self.predict(X), axis=1)  
        sum =0
        for i in range(n):
            if E[i]!=0:
                sum += 1 
        return sum/n
    
    def loss(self,X,Y):
        n_l = X.shape[0]
        P = np.zeros((n_l,self.k))
        
        for i in range(n_l):
            for j in range(self.k):
                P[i,j]= proba(j, X[i,:], self.coeff)
        
        return - np.sum(np.log(P) *Y)/X.shape[0] 

def proba(a, x, coeff):
    return np.exp(x.dot(coeff[:,a]))/np.sum(np.exp(x.dot(coeff)))    

