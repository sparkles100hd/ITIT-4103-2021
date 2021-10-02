import math
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

def appendedX(degree, X):
    Xappended = X
    for i in range(2, degree + 1): #it starts with 2 since for deg>=2 we dont want redundant 1st column
        extrafeat = (X[:, 1] ** i).reshape(Xappended.shape[0], 1)
        Xappended = np.append(Xappended, extrafeat,axis=1)

    return Xappended

X=np.random.random((50,1))
Ytrain=np.sin(1 + np.square(X))
noise=np.random.normal(0,0.032,size=(50,1))
Ynoise= Ytrain + noise
train=40 #8:2 = 4:1 = 40:10 for 50 points

Xtest=X[train:]
Ytest=Ynoise[train:]
Xtrain=X[:train]

A = np.append(np.ones((Xtrain.shape[0], 1)), Xtrain, axis=1) # appended ones for training set
B= np.append(np.ones((Xtest.shape[0], 1)), Xtest, axis=1) # appended ones for testing set
Ytrain = Ynoise[:train]

indstrain=Xtrain.argsort(axis=0) #sorting the data for plotting (plot doesnt work properly otherwise)
Xtrainreal=Xtrain[indstrain].reshape(Xtrain.shape[0], 1)
Ytrainreal=Ytrain[indstrain].reshape(Ytrain.shape[0], 1)

indstest=Xtest.argsort(axis=0)
Xtestreal=Xtest[indstest].reshape(Xtest.shape[0], 1)
Ytestreal=Ytest[indstest].reshape(Ytest.shape[0], 1)


degree = 1
X1 = appendedX(degree, A) #getting the X matrix with added features
W1 = inv(X1.T @ X1) @ X1.T @ Ytrain  #getting the coefficient matrix

X1=X1[X1[:,1].argsort()] # sorting
Y1=X1@W1# predicting train data

X1t=appendedX(degree, B)#getting X matrix for test input
X1t=X1t[X1t[:,1].argsort()] # sorting
Y1t=X1t@W1# predicting test data

plt.subplot(4,2,1)
plt.title('Degree 1 on Train Data')
plt.plot(Xtrainreal, Ytrainreal,label='real')
plt.legend(loc='best')
plt.plot(X1[:,1],Y1 ,c='r',label='predicted')
plt.legend(loc='best')

plt.subplot(4,2,2)
plt.title('Degree 1 on Test Data')
plt.plot(Xtestreal, Ytestreal,label='real')
plt.legend(loc='best')
plt.plot(X1t[:,1] ,Y1t,c='r',label='predicted')
plt.legend(loc='best')

RMSE1 = math.sqrt(np.square(np.subtract(Ytest, Y1t)).mean())
RMSE2 = math.sqrt(np.square(np.subtract(Ytrain, Y1)).mean())
print('RMSE of Test Set Degree 1- ',RMSE1)
print('RMSE of Train Set Degree 1- ',RMSE2,'\n')

##########################################
degree = 2
X2 = appendedX(degree, A)
W2 = inv(X2.T @ X2) @ X2.T @ Ytrain
X2=X2[X2[:,1].argsort()] # sorting
Y2=X2@W2# predicting train data

X2t=appendedX(degree, B)
X2t=X2t[X2t[:,1].argsort()] # sorting
Y2t=X2t@W2# predicting train data

plt.subplot(4,2,3)
plt.title('Degree 2 on Train Data')
plt.plot(Xtrainreal, Ytrainreal,label='real')
plt.legend(loc='best')
plt.plot(X2[:,1],Y2 ,c='r',label='predicted')
plt.legend(loc='best')

plt.subplot(4,2,4)
plt.title('Degree 2 on Test Data')
plt.plot(Xtestreal, Ytestreal,label='real')
plt.legend(loc='best')
plt.plot(X2t[:,1] ,Y2t,c='r',label='predicted')
plt.legend(loc='best')

RMSE1 = math.sqrt(np.square(np.subtract(Ytest, Y2t)).mean())
RMSE2 = math.sqrt(np.square(np.subtract(Ytrain, Y2)).mean())
print('RMSE of Test Set Degree 2- ',RMSE1)
print('RMSE of Train Set Degree 2- ',RMSE2,'\n')
######################################
degree = 3
X3 = appendedX(degree, A)
W3 = inv(X3.T @ X3) @ X3.T @ Ytrain
X3=X3[X3[:,1].argsort()] # sorting
Y3=X3@W3# predicting train data

X3t=appendedX(degree, B)
X3t=X3t[X3t[:,1].argsort()] # sorting
Y3t=X3t@W3# predicting train data

plt.subplot(4,2,5)
plt.title('Degree 3 on Train Data')
plt.plot(Xtrainreal, Ytrainreal,label='real')
plt.legend(loc='best')
plt.plot(X3[:,1],Y3 ,c='r',label='predicted')
plt.legend(loc='best')

plt.subplot(4,2,6)
plt.title('Degree 3 on Test Data')
plt.plot(Xtestreal, Ytestreal,label='real')
plt.legend(loc='best')
plt.plot(X3t[:,1] ,Y3t,c='r',label='predicted')
plt.legend(loc='best')

RMSE1 = math.sqrt(np.square(np.subtract(Ytest, Y3t)).mean())
RMSE2 = math.sqrt(np.square(np.subtract(Ytrain, Y3)).mean())
print('RMSE of Test Set Degree 3- ',RMSE1)
print('RMSE of Train Set Degree 3- ',RMSE2,'\n')
##################################
degree = 4
X4 = appendedX(degree, A)
W4 = inv(X4.T @ X4) @ X4.T @ Ytrain
X4=X4[X4[:,1].argsort()] # sorting
Y4=X4@W4# predicting train data

X4t=appendedX(degree, B)
X4t=X4t[X4t[:,1].argsort()] # sorting
Y4t=X4t@W4# predicting train data

plt.subplot(4,2,7)
plt.title('Degree 4 on Train Data')
plt.plot(Xtrainreal, Ytrainreal,label='real')
plt.legend(loc='best')
plt.plot(X4[:,1],Y4 ,c='r',label='predicted')
plt.legend(loc='best')

plt.subplot(4,2,8)
plt.title('Degree 4 on Test Data')
plt.plot(Xtestreal, Ytestreal,label='real')
plt.legend(loc='best')
plt.plot(X4t[:,1] ,Y4t,c='r',label='predicted')
plt.legend(loc='best')

RMSE1 = math.sqrt(np.square(np.subtract(Ytest, Y4t)).mean())
RMSE2 = math.sqrt(np.square(np.subtract(Ytrain, Y4)).mean())
print('RMSE of Test Set Degree 4- ',RMSE1)
print('RMSE of Train Set Degree 4- ',RMSE2,'\n')

plt.tight_layout()
plt.show()