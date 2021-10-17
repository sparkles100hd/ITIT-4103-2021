import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading and storing the data
data = pd.read_csv('ex1data1.txt', header = None).values
x=data[:,0]
y=data[:,1]

#plotting data points
plt.subplot(1,4,1)
plt.title('Population(x) vs Profit(y)')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.scatter(x,y,color='red',label='data points')
plt.legend(loc='best')

#our parameters and variables
alpha=0.1
theta0=0
theta1=0
noOfIteration=2

#printing intial cost function
h=(theta0*x) + theta1
print('learning rate -',alpha)
print('initial cost 0 steps - ',np.sum((y-h)*(y-h))/(2*len(x)))


#the gradient descent implementation
for i in range(1,noOfIteration):
    h=(theta0*x) + theta1
    theta0-= alpha * ((-1.0/len(x)) * (np.sum(x*(y-h))))
    theta1-= alpha * ((-1.0/len(x))* (np.sum(y-h)))


#printing final cost value (for checking personally)
h=(theta0*x) + theta1
print('Final cost 2000 steps - ',np.sum((y-h)*(y-h))/(2*len(x)))
theta1Final=theta1
theta0Final=theta0


#sorting the scatter points and the fit model points. sorting to get a proper plt graph else it wont work
ind=x.argsort(axis=0)
x=x[ind].reshape(x.shape[0],1)
h=h[ind].reshape(h.shape[0],1)
y=y[ind].reshape(y.shape[0],1)


#plotting the model vs actual data
plt.subplot(1,4,2)
plt.title('Model vs Actual')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.scatter(x,y,color='red',label='data points')
plt.plot(x,h,color='green',label='fit model')
plt.legend(loc='best')

#generating points for contour plot
theta0 = np.linspace(-5, 5, 50) #c
theta1 = np.linspace(-10, 10, 50) #m

#creating 2d cost function matrix (Z matrix)
costFunction = np.zeros((theta1.shape[0], theta0.shape[0]))

for i, teta1 in enumerate(theta1):
    for j, teta0 in enumerate(theta0):
        h = (teta0 * x) + teta1
        costFunction[i, j] = np.sum((y-h)*(y-h))/(2.0*len(x)) #Z Matrix for each (theta1,theta0)

plt.subplot(1,4,3)
plt.xlabel('theta1')
plt.ylabel('theta0')
plt.title('Contour')
plt.scatter(theta1Final, theta0Final,label='Trained parameter')
plt.colorbar(plt.contour(theta1, theta0, costFunction.T,levels=np.linspace(0,500,10)))
plt.legend(loc='best')
plt.suptitle('Learning Rate - ' + str(alpha))


plt.subplot(1,4,4)
plt.xlabel('theta1')
plt.ylabel('theta0')
plt.title('Contour')
plt.scatter(theta1Final, theta0Final,label='Trained parameter')
plt.colorbar(plt.contourf(theta1, theta0, costFunction.T,levels=np.linspace(0,500,10)))
plt.legend(loc='best')
plt.suptitle('Learning Rate - ' + str(alpha))
plt.show()