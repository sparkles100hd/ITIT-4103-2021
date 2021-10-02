import numpy as np
import matplotlib.pyplot as plt

X=np.random.random((50,1))
Y=np.sin(1+np.square(X))
noise=np.random.normal(0,0.032,size=(50,1))
Ynoise=Y+noise
train=40 #8:2 = 4:1 = 40:10 for 50 points

plt.suptitle("NoiseFull and True Function")

Xtest=X[train:]
Ytest=Y[train:]
Yntest=Ynoise[train:]

Xtrain=X[:train]
Ytrain=Y[:train]
Yntrain=Ynoise[:train]

traini=Xtrain.argsort(axis=0) #sorting for plt plot
testi=Xtest.argsort(axis=0)

Xtest=Xtest[testi].reshape(10,1)
Ytest=Ytest[testi].reshape(10,1)
Yntest=Yntest[testi].reshape(10,1)

Xtrain=Xtrain[traini].reshape(40,1)
Ytrain=Ytrain[traini].reshape(40,1)
Yntrain=Yntrain[traini].reshape(40,1)

plt.subplot(2,2,1)
plt.title("Training plot")
plt.plot(Xtrain,Ytrain,label="true function")
plt.plot(Xtrain,Yntrain,label="y")
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.title("Testing plot")
plt.plot(Xtest,Ytest,label="true function")
plt.plot(Xtest,Yntest,label="y")
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.title("Training scatter")
plt.scatter(Xtrain,Ytrain,label="true function")
plt.scatter(Xtrain,Yntrain,label="y")
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.title("Testing scatter")
plt.scatter(Xtest,Ytest,label="true function")
plt.scatter(Xtest,Yntest,label="y")
plt.legend(loc='best')

plt.show()
