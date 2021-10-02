import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math

X=np.random.random((50,1))
Y=np.sin(1+np.square(X))
noise=np.random.normal(0,0.032,size=(50,1))
Ynoise=Y+noise
train=40

Xtest=X[train:]
Ytest=Ynoise[train:]
Xtrain=X[:train]
Ytrain=Ynoise[:train]

ind=Xtrain.argsort(axis=0)
Xtrain=Xtrain[ind].reshape(Xtrain.shape[0],1)
Ytrain=Ytrain[ind].reshape(Ytrain.shape[0],1)

ind=Xtest.argsort(axis=0)
Xtest=Xtest[ind].reshape(Xtest.shape[0],1)
Ytest=Ytest[ind].reshape(Ytest.shape[0],1)

A = np.concatenate((np.ones((Xtrain.shape[0], 1)),Xtrain), axis=1) # concatenates the '1' column to train
W = inv(A.T @ A) @ A.T @ Ytrain

y1 = W[1] * Xtest + W[0]
y2= W[1] * Xtrain + W[0]

plt.subplot(2,2,1)
plt.title("prediction on test data scatter")
plt.scatter(Xtest, y1,label='predicted')
plt.scatter(Xtest, Ytest,label='original')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.title("prediction on train data scatter")
plt.scatter(Xtrain, y2,label='predicted')
plt.scatter(Xtrain, Ytrain,label='original')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.title("prediction on test data plot")
plt.plot(Xtest, y1,label='predicted')
plt.plot(Xtest, Ytest,label='original')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.title("prediction on train data plot")
plt.plot(Xtrain, y2,label='predicted')
plt.plot(Xtrain, Ytrain,label='original')
plt.legend(loc='best')

RMSE1 = math.sqrt(np.square(np.subtract(Ytest, y1)).mean())
RMSE2 = math.sqrt(np.square(np.subtract(Ytrain, y2)).mean())

print('RMSE of Test Set - ',RMSE1)
print('RMSE of Train Set - ',RMSE2)

plt.show()



