import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Euclidean:
    '''
    Implementation of Euclidean metric
    '''
    def __init__(self):
        '''
        Instantiate Eucliden metric

        Args: None
        '''
        pass
    
    def distance(self, x1, x2):
        '''
        Euclidean distance between x1 and x2

        Args: x1 (vec): first point,  x2 (vec): second point
        '''
        return np.sqrt(np.sum((x1-x2)**2))
    

    def metric_aa(self, X):
        '''
        Calculation of cross distances of X instances

        Args:
            X (list): list of instances
        '''

        r = len(X) # number of rows (instances) in X
        distances = np.zeros(shape=(r,r), dtype=np.float32) # output matrix
        
        # triangular calculation of distances
        for i in range(r-1):
            for j in range(i+1, r):
                aux = self.distance(X[i], X[j]) # d(i,j) = d(j,i)
                distances[i,j] = aux
                distances[j,i] = aux
        
        return distances


    def metric_ab(self, X1, X2):
        '''
        Calculation of distances of points in X1 with points in X2

        Args:
            X1 (list): first list of instances
            X2 (list): second list of instances
        '''

        r1 = len(X1) # number of rows (instances) in X1
        r2 = len(X2) # number of rows (instances) in X2
        distances = np.zeros(shape=(r1,r2), dtype=np.float32) # output matrix
        
        # calculation of distances
        for i in range(r1):
            for j in range(r2):
                distances[i,j] = self.distance(X1[i], X2[j])
        
        return distances

                


class kTLNN:
    def __init__(self, k, kb_factor=1.4):
        self.NN1st = []
        self.NN2nd = []
        self.k = k
        self.kb_factor =  kb_factor
        self.metric = Euclidean()
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.m1, self.n1 = X_train.shape # m1 vectores de entrenamiento con n caracteristicas
        self.tags = np.unique(self.y_train) # lista de clases

        self.Dmatrix_train = self.metric.metric_aa(self.X_train) # calculo de las distancias en X_train
    
    def metric_full(self, X_test):
        trtr = self.Dmatrix_train # Training set matrix
        tete = self.metric.metric_aa(X_test) # Test set matrix
        trte = self.metric.metric_ab(self.X_train, X_test)
        tetr = np.transpose(trte)

        # Concat both matrices
        trtr = np.concatenate([trtr, tetr], axis=0)
        tete = np.concatenate([trte, tete], axis=0)

        return np.concatenate([trtr, tete], axis=1)
    
    def predict(self, X_test):
        self.Dmatrix = self.metric_full(X_test)
        predictions = []
        for i, query in enumerate(X_test):
            self.getNN1st(self.m1+i)
            self.getNN2nd(self.m1+i)
            self.getNNext(query)
            self.getNNtwo(self.m1+i)
            y = self.votes()
            predictions.append(y)
        return predictions

    def _calculatekNN(self, x_i, k): # x_i es un indice
        distances = self.Dmatrix[x_i, 0:self.m1]
        k_i = np.argsort(distances)[:int(k)]

        return k_i

    def getNN1st(self, x_qi):
        self.NN1st = []
        self.NN1st = self._calculatekNN(x_qi, self.k) # INDICES
        self.distances = [self.Dmatrix[x_qi, i] for i in self.NN1st]

    def getNN2nd(self, x_qi):
        self.NN2nd = []

        k_i = [self._calculatekNN(i, self.k+1) for i in self.NN1st]
        NN2nd_aux = [j[1:] for j in k_i]
    
        firstRatio = self.Dmatrix[x_qi, self.NN1st[-1]]

        for idx, k in enumerate(NN2nd_aux):
            dist = [self.Dmatrix[x_qi, j] for j in k]
            k_eff_i = np.where(dist < 2*firstRatio)
            k = k[k_eff_i[0]]
            k = np.insert(k, 0, self.NN1st[idx], axis = 0) # insertamos el nn1st para el calculo de centroides (indice 0)

            self.NN2nd.append(k)
    
    def centroids(self): # centroide promedio
        centroids_ = [np.sum(self.X_train[i], axis = 0)/len(i) for i in self.NN2nd]
        return centroids_

    def getNNext(self, x_q):
        centroids = self.centroids()
        dist = [self.metric.distance(x_q, j) for j in centroids] # CHECAR
        self.NNext = []
        
        for i in range(len(dist)):
            if dist[i]<self.distances[i]:
                for j in self.NN2nd[i]:
                    self.NNext.append(j)            
            else:
                self.NNext.append(self.NN1st[i])
        self.NNext = np.array(self.NNext)

    def backwards_knn(self, x_i, x_qi, kb):
        distances = self.Dmatrix[x_i, 0:self.m1]
        distances = np.concatenate((distances, [self.Dmatrix[x_i, x_qi]]))

        k_i = np.argsort(distances)[:int(kb+1)]
        
        return len(distances)-1 in k_i

    def getNNtwo(self, x_qi):
        kb = np.ceil(self.kb_factor*self.k)
        self.NNext = np.unique(self.NNext, axis = 0)
        self.NNtwo = []

        for ni in self.NNext:
            if self.backwards_knn(ni, x_qi, kb):
                self.NNtwo.append(ni)
        self.NNtwo = np.array(self.NNtwo)

    def votes(self):
        if len(self.NNtwo) == 0: # considera vecindarios vacios
            k_labels = [self.y_train[label] for label in self.NN1st]
        else:
            k_labels = [self.y_train[label] for label in self.NNtwo]
        values, counts = np.unique(k_labels, return_counts = True)
        ind = np.argmax(counts)
        tag = values[ind]
        return tag

    
if __name__ == '__main__':   
    dataset = pd.read_csv('Datasets/iris.csv')
    
    X = dataset.drop('class', axis=1).values
    y = dataset['class'].values

    # dividimos datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) #12

    classifier = kTLNN(k=11)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    print(f'Preds: {preds}')
    print(f'y_test: {y_test}')
    print(f'accuracy: {accuracy_score(y_true=y_test, y_pred=preds)}')