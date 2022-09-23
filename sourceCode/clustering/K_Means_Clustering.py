import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

#K-Means Clustering
def kMeansClustering(Xtest, Xtrain, Ytest, Ytrain, dataSet):

    # Formatting the dataSets for analysis
    XTrain = Xtrain[:,:].transpose()[1:,:].tolist()[0]
    XTest = Xtest[:,:].transpose()[1:,:].tolist()[0]
    YTrain = Ytrain[:,:].transpose()[1:,:].tolist()[0]
    YTest = Ytest[:,:].transpose()[1:,:].tolist()[0]


    mpl.rcParams['figure.figsize'] = [5, 5]

    X_combined = np.r_[XTrain, XTest]
    Y_combined = np.r_[YTrain, YTest]

    # Manually casting to int 
    X_combined = np.array(X_combined, dtype='int')
    Y_combined = np.array(Y_combined, dtype='int')

    # Within Cluster sum of Squares
    wcss = []
    for i in range(1, 11):
        kMeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kMeans.fit(X_combined.reshape(-1,1))
        wcss.append(kMeans.inertia_)

    plt.plot(range(1,11), wcss)
    plt.title(f'K-Means Clustering for {dataSet}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x 