import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from matplotlib import pyplot as bar

count =0

class NaiveBayes(object):
    def fit(self, X, Y, smoothing=10e-4):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]

            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var,allow_singular=True) + np.log(self.priors[c])

        return np.argmax(P, axis=1)


if __name__ == '__main__':
    data = pd.read_csv('iris.data', header=None).values
    np.random.shuffle(data)#shuffle the data otherwise we get bad result
    X = data[:, :4]
    Y = data[:, 4:]
    flower=['SepalLength','SepalWidth','PetalLength','PetalWidth']
    classes=['Iris-setosa','Iris-versicolor','Iris-virginica']

    for i,num in enumerate(Y):
        if Y[i]=='Iris-setosa':
            Y[i]=0
        if Y[i]=='Iris-versicolor':
            Y[i]=1
        if Y[i]=='Iris-virginica':
            Y[i]=2

    Y=Y.flatten()#2d array to 1d vector
    Ntrain = int(0.6*len(Y))#60% of samples for train
    Xtrain, Ytrain = X[:int(Ntrain)], Y[:int(Ntrain)]#60% train
    Xtest, Ytest = X[int(Ntrain):], Y[int(Ntrain):]#rest 40% test

    visit=[0,0,0,0]
    Xfeat=np.zeros((X.shape[0],1))
    totalaccuracy,maxaccuracy,ind=0,0,0

    model = NaiveBayes()
    model.fit(Xtrain, Ytrain)
    FIN = model.predict(Xtest)

    used,accuracy,ystr=[],[],[]

    for i in range(0,4):
        count = 1
        ti=i
        for j in range(0,4):
            if visit[j]==1:
                continue

            #adding the feature to test to our feature set and training and testing our model
            model = NaiveBayes()
            Xfeat=np.append(Xfeat,np.reshape(X[:,j],(len(X[:,0]),1)),axis=1)
            model.fit(Xfeat[:Ntrain,1:], Ytrain)

            #after training on the new feature set we test our model and note accuracy
            P = model.predict(Xfeat[Ntrain:,1:])
            tac=np.mean(P==Ytest)
            accuracy.append(tac)

            # PLOTTING BLOCK IGNORE (TO PLOT THE INSTANCES)
            if(i==0):
                C0,C0i,C1,C1i,C2,C2i,=[],[],[],[],[],[]

                for l,num in enumerate(Ytest):
                    if num==P[l]:
                        if num==0:C0.append(Xfeat[l,1])
                        elif num==1:C1.append(Xfeat[l, 1])
                        elif num==2:C2.append(Xfeat[l,1])
                    else:
                        if num==0:C0i.append(Xfeat[l,1])
                        elif num==1:C1i.append(Xfeat[l, 1])
                        elif num==2:C2i.append(Xfeat[l,1])

                plt.suptitle('All instances of each class from selected feature')
                plt.subplot(4,3,count)
                plt.title(str(flower[j])+' '+str(classes[0]))
                plt.scatter(C0,C0,color='green')
                plt.scatter(C0i, C0i,color='red')
                count += 1
                plt.subplot(4, 3, count)
                plt.title(str(flower[j])+' '+str(classes[1]))
                plt.scatter(C1,C1,color='green')
                plt.scatter(C1i, C1i,color='red')
                count += 1
                plt.subplot(4, 3, count)
                plt.title(str(flower[j])+' '+str(classes[2]))
                plt.scatter(C2,C2,color='green')
                plt.scatter(C2i, C2i,color='red')
                count += 1

            # PLOTTING BLOCK IGNORE (TO PLOT THE INSTANCES)
            elif (i == 1):
                C0x,C0y, C0xi,C0yi, C1x,C1y, C1xi,C1yi,C2x,C2y, C2xi,C2yi=[],[],[],[],[],[],[],[],[],[],[],[]

                for l, num in enumerate(Ytest):
                    if num == P[l]:
                        if num == 0:
                            C0x.append(Xfeat[l, 1])
                            C0y.append(Xfeat[l, 2])
                        elif num == 1:
                            C1x.append(Xfeat[l, 1])
                            C1y.append(Xfeat[l, 2])
                        elif num == 2:
                            C2x.append(Xfeat[l, 1])
                            C2y.append(Xfeat[l, 2])
                    else:
                        if num == 0:
                            C0xi.append(Xfeat[l, 1])
                            C0yi.append(Xfeat[l, 2])
                        elif num == 1:
                            C1xi.append(Xfeat[l, 1])
                            C1yi.append(Xfeat[l, 2])
                        elif num == 2:
                            C2xi.append(Xfeat[l, 1])
                            C2yi.append(Xfeat[l, 2])

                plt.suptitle('All instances of each class from selected 2 feature')
                plt.subplot(4, 3, count)
                plt.title(classes[0])
                plt.xlabel(str(flower[used[0]]))
                plt.ylabel(str(flower[j]))
                plt.scatter(C0x, C0y, color='green')
                plt.scatter(C0xi, C0yi, color='red')
                count += 1
                plt.subplot(4, 3, count)
                plt.title(classes[1])
                plt.xlabel(str(flower[used[0]]))
                plt.ylabel(str(flower[j]))
                plt.scatter(C1x, C1y, color='green')
                plt.scatter(C1xi, C1yi, color='red')
                count += 1
                plt.subplot(4, 3, count)
                plt.title(classes[2])
                plt.xlabel(str(flower[used[0]]))
                plt.ylabel(str(flower[j]))
                plt.scatter(C2x, C2y, color='green')
                plt.scatter(C2xi, C2yi, color='red')
                count += 1

            #storing the feature set as strings so we can show it in bar plot
            if ti==0:ystr.append(str(flower[j]))
            elif ti==1:ystr.append(str(flower[used[0]])+','+str(flower[j]))
            elif ti==2:ystr.append(str(flower[used[0]])+','+str(flower[used[1]])+','+str(flower[j]))
            elif ti==3:ystr.append(str(flower[used[0]]) + ',' + str(flower[used[1]]) + ',' + str(flower[used[2]]) + ','+ str(flower[j]))

            #basically tracks the feature which on adding gives a max accuracy on the model compared to others
            if(tac>maxaccuracy):
                maxaccuracy=tac
                ind=j

            #delete the added column after use so we can add next feature and test
            Xfeat=np.delete(Xfeat, -1, axis=1)

        used.append(ind)#add the feature that is the best
        Xfeat=np.append(Xfeat,np.reshape(X[:,ind],(len(X[:,0]),1)),axis=1)#appending that feature that gave us max accuracy
        visit[ind]=1#marking that we visited and used the best feature to our subset
        maxaccuracy=0

        if i==0 or i==1:#showing the plots for 1 feature, 2 feature set
            plt.tight_layout()
            plt.show()


    #plotting the accuracy bar plot
    accuracy.append(np.mean(FIN==Ytest))
    ystr.append('0,1,2,3')
    y_pos = np.arange(len(ystr))
    bar.barh(y_pos,accuracy,color='red',linewidth=2)
    bar.yticks(y_pos, ystr)
    for index, value in enumerate(accuracy):
        bar.text(value, index, str(value))
    bar.tight_layout()
    bar.show()

