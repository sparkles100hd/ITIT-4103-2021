from sklearn.cluster import KMeans
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
v = np.vectorize(lambda x: colors[x % len(colors)])

#input iris dataset and format and store in our variables
data = pd.read_csv('iris.data', header=None).values
np.random.shuffle(data)  # shuffle the data otherwise we get bad result
X = data[:, :4]
Y = data[:, 4:]
flower = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#change class label to numeric
for i, num in enumerate(Y):
    if Y[i] == 'Iris-setosa':
        Y[i] = 0
    if Y[i] == 'Iris-versicolor':
        Y[i] = 1
    if Y[i] == 'Iris-virginica':
        Y[i] = 2

Y = Y.flatten()  # 2d array to 1d vector

#KMEANS Algorithm
model1=KMeans(n_clusters=3)
model1.fit(X)
YKM=model1.labels_
print('K Means Accuracy - ',np.mean(YKM==Y))
plt.title('K Means')

#EM Algorithm
model2=GaussianMixture(n_components=3)
model2.fit(X)
YEM= model2.predict(X)
print('GMM EM Accuracy - ',np.mean(YEM==Y),'\n')


#PCA Algorithm. first we show the variance
pca = PCA()
pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_
print('PCA Explained variance ratio - ',explained_variance,'\n')

#PCA Algorithm. we train for first 2 principal components directly.
pca=PCA(n_components=2)
XN=pca.fit_transform(X)

#K Means on our PCA Output
model3=KMeans(n_clusters=3,init='k-means++')
model3.fit(XN)
YKMPCA=model3.labels_
print('K Means PCA Accuracy - ',np.mean(YKMPCA==Y))

#EM on our PCA Output
model4=GaussianMixture(n_components=3)
model4.fit(XN)
YEMPCA= model4.predict(XN)
print('GMM EM PCA Accuracy - ',np.mean(YEMPCA==Y))


#plot block
plt.title('Results')
plt.subplot(2,3,1)
plt.title('Original')
plt.scatter(X[:,0], X[:,1],c=v(Y))
plt.subplot(2,3,2)
plt.title('K Means')
plt.scatter(X[:,0], X[:,1],c=v(YKM))
plt.scatter(model1.cluster_centers_[:,0],model1.cluster_centers_[:,1],color='black')
plt.subplot(2,3,3)
plt.title('K Means PCA')
plt.scatter(X[:,0], X[:,1],c=v(YKMPCA))
plt.subplot(2,3,4)
plt.title('Original')
plt.scatter(X[:,0], X[:,1],c=v(Y))
plt.subplot(2,3,5)
plt.title('GMM EM')
plt.scatter(X[:,0], X[:,1],c=v(YEM))
plt.subplot(2,3,6)
plt.title('GMM EM PCA')
plt.scatter(X[:,0], X[:,1],c=v(YEMPCA))
plt.show()
