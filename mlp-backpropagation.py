import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt


np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def split(X,T):
    # 1
    Xval0 = np.vstack([X[0:10],X[50:60],X[100:110]])
    Xtrain0 = np.vstack([X[10:50],X[60:100],X[110:150]])
    Tval0 = np.vstack([T[0:10],T[50:60],T[100:110]])
    Ttrain0 = np.vstack([T[10:50],T[60:100],T[110:150]])
    # 2
    Xval1 = np.vstack([X[10:20],X[60:70],X[110:120]])
    Xtrain1 = np.vstack([X[0:10],X[20:50],X[50:60],X[70:100],X[100:110],X[120:150]])
    Tval1 = np.vstack([T[10:20],T[60:70],T[110:120]])
    Ttrain1 = np.vstack([T[0:10],T[20:50],T[50:60],T[70:100],T[100:110],T[120:150]])
    # 3
    Xval2 = np.vstack([X[20:30],X[70:80],X[120:130]])
    Xtrain2 = np.vstack([X[0:20],X[30:50],X[50:70],X[80:100],X[100:120],X[130:150]])
    Tval2 = np.vstack([T[20:30],T[70:80],T[120:130]])
    Ttrain2 = np.vstack([T[0:20],T[30:50],T[50:70],T[80:100],T[100:120],T[130:150]])
    # 4
    Xval3 = np.vstack([X[30:40],X[80:90],X[130:140]])
    Xtrain3 = np.vstack([X[0:30],X[40:50],X[50:80],X[90:100],X[100:130],X[140:150]])
    Tval3 = np.vstack([T[30:40],T[80:90],T[130:140]])
    Ttrain3 = np.vstack([T[0:30],T[40:50],T[50:80],T[90:100],T[100:130],T[140:150]])
    # 5
    Xval4 = np.vstack([X[40:50],X[90:100],X[140:150]])
    Xtrain4 = np.vstack([X[0:40],X[50:90],X[100:140]])
    Tval4 = np.vstack([T[40:50],T[90:100],T[140:150]])
    Ttrain4 = np.vstack([T[0:40],T[50:90],T[100:140]])

    return Xtrain0,Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xval0,Xval1,Xval2,Xval3,Xval4,Ttrain0,Ttrain1,Ttrain2,Ttrain3,Ttrain4,Tval0,Tval1,Tval2,Tval3,Tval4


def kfold(Xtrain0,Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xval0,Xval1,Xval2,Xval3,Xval4,Ttrain0,Ttrain1,Ttrain2,Ttrain3,Ttrain4,Tval0,Tval1,Tval2,Tval3,Tval4,counter):
    while counter < 5:
        if counter == 0:
            Xtrain = Xtrain0
            Xval = Xval0
            Ttrain =Ttrain0
            Tval = Tval0
        if counter == 1:
            Xtrain = Xtrain1
            Xval = Xval1
            Ttrain =Ttrain1
            Tval = Tval1
        if counter == 2:
            Xtrain = Xtrain2
            Xval = Xval2
            Ttrain =Ttrain2
            Tval = Tval2
        if counter == 3:
            Xtrain = Xtrain3
            Xval = Xval3
            Ttrain =Ttrain3
            Tval = Tval3
        if counter == 4:
            Xtrain = Xtrain4
            Xval = Xval4
            Ttrain =Ttrain4
            Tval = Tval4
        counter += 1

    if counter == 5:
        counter = 0
    
    return Xtrain, Xval, Ttrain, Tval

def train_and_validate(X,T,W1,b1,W2,b2,counter,epochs):
    Xtrain0,Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xval0,Xval1,Xval2,Xval3,Xval4,Ttrain0,Ttrain1,Ttrain2,Ttrain3,Ttrain4,Tval0,Tval1,Tval2,Tval3,Tval4 = split(X,T)

    for epoch in range(epochs):
        ## KFOLD
        Xtrain, Xval, Ttrain, Tval = kfold(Xtrain0,Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xval0,Xval1,Xval2,Xval3,Xval4,Ttrain0,Ttrain1,Ttrain2,Ttrain3,Ttrain4,Tval0,Tval1,Tval2,Tval3,Tval4,counter)

        ## TRAIN
        # forward pass
        A = sigmoid(Xtrain.dot(W1) + b1) # A = sigma(Z)
        Y = softmax(A.dot(W2) + b2) # Y = softmax(Z2)
        #Y = sigmoid(A.dot(W2) + b2) # Y = softmax(Z2)

        # backward pass
        delta2 = Y - Ttrain
        delta1 = (delta2).dot(W2.T) * A * (1 - A)

        W2 -= alpha * A.T.dot(delta2)
        b2 -= alpha * (delta2).sum(axis=0)

        W1 -= alpha * Xtrain.T.dot(delta1)
        b1 -= alpha * (delta1).sum(axis=0)

        # save loss function values across training iterations
        #loss = np.average(-Ttrain * np.log(Y))
        loss = np.average((Ttrain - Y)**2)
        #print('Loss function value: ', loss)
        costsTrain.append(loss)

        ## validate
        # forward pass
        Aval = sigmoid(Xval.dot(W1) + b1)
        Yval = softmax(Aval.dot(W2) + b2)

        #loss = np.average(-Tval * np.log(Yval))
        loss = np.average((Tval - Yval)**2)
        #print('Loss function value: ', loss)
        costsVal.append(loss)
    

# data input
X = genfromtxt('iris.txt', delimiter=',')
X = np.delete(X, 4, 1)  # delete label


# generate one-hot-encodings
labels = np.array([0]*50 + [1]*50 + [2]*50)
T = np.zeros((150, 3))
for i in range(150):
    T[i, labels[i]] = 1


# var
samples = X.shape[0] # 150 samples
features = X.shape[1] # 4 features
hidden_nodes = 5
classes = 3
alpha = 10e-4
counter = 0
costsTrain = []
costsVal = []

# initial weights
W1 = np.random.randn(features, hidden_nodes)
b1 = np.random.randn(hidden_nodes)
W2 = np.random.randn(hidden_nodes, classes)
b2 = np.random.randn(classes)

train_and_validate(X,T,W1,b1,W2,b2,counter,100)

# Plot
fig = plt.figure()
fig.suptitle('MSE Graph')
sns.set_style("darkgrid")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(costsTrain, label = "Train")
plt.plot(costsVal, label = "Val")
plt.legend(loc='best')
plt.show()
