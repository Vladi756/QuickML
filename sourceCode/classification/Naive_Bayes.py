import matplotlib as mpl

mpl.use("Agg")

from telnetlib import GA
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# Importing confusion_matrix function
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Naive Bayes
def Naive_Bayes(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    '''
    Takes pre processed data and the dataSet which expects the algorithm
    to be placed on its data. Performs Naive Bayes classification
    on the given dataset and returns a confusion matrix to display the accuracy
    and loss of the ML model. The classification is based on Bayes theoreum.
    '''
    
    # Formatting the dataSets for analysis
    XTrain = np.array(Xtrain[:,:].transpose()[1:,:].tolist()[0])
    XTest = np.array(Xtest[:,:].transpose()[1:,:].tolist()[0])
    YTrain = np.array(Ytrain[:,:].transpose()[1:,:].tolist()[0])
    YTest = np.array(Ytest[:,:].transpose()[1:,:].tolist()[0])

    # Manually casting to int using labelEncoder class to preserve
    # data integrity. (better than .astype('int'))
    lab_enc = preprocessing.LabelEncoder()

    XTrain = lab_enc.fit_transform(XTrain)
    XTest = lab_enc.fit_transform(XTest)
    YTrain = lab_enc.fit_transform(YTrain)
    YTest = lab_enc.fit_transform(YTest)

    # Creating Naive Bayes Classifier
    classifier = GaussianNB()

    # Fit classifier to training set
    classifier.fit(XTrain.reshape(-1,1), YTrain.reshape(-1,1))

    YPred = classifier.predict(XTest.reshape(-1,1))

    # Making the Confusion Matrix
    cm = confusion_matrix(YTest, YPred)
 
    # Plotting the Confusion Matrix 
    plot_confusion_matrix(classifier, XTest.reshape(-1,1), YTest.reshape(-1,1))

    plt.title(f'Naive Bayes Classification for {dataSet}')

    plt.legend() 
    
    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x