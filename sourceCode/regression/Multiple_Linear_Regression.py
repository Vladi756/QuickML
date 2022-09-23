import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib as mpl
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# MULTIPLE LINEAR REGRESSION 
def multipleLinearRegression(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    '''
    Takes pre processed data and the dataSet which expects the algorithm
    to be placed on its data. It saves the graph as a figure and returns it 
    to be later displayed in the html.  
    '''

    # Fitting Multiple Linear Regression to Training Set 
    regressor = LinearRegression()
    regressor.fit(Xtrain, Ytrain)

    # Test set prediction 
    Ypred = regressor.predict(Xtest)
    Ypred2 = regressor.predict(Xtrain)

    plt.title(f'Multivariate Linear Regression for Dataset: {dataSet}')

     # Adding train and test plot 
    train_plot = plt.subplot(121)
    test_plot = plt.subplot(122)
    # Setting size of figure
    mpl.rcParams['figure.figsize'] = [10, 10]

    # Formatting the predictions for plotting
    YTest_Hat_Plot = Ypred.transpose()[1:,:].tolist()[0]
    YTrain_Hat_Plot = Ypred2.transpose()[1:,:].tolist()[0]

    # Train set Plotted 
    train_plot.grid(True)
    train_plot.set_title('Train Set')

    # Scattering Actual Train Set
    train_plot.scatter(Xtrain[:,:].transpose()[1:,:].tolist()[0], 
                       Ytrain[:,:].transpose()[1:,:].tolist()[0],
                       color ='Orange',
                       label = 'Actual Train Set')    

    # Scattering Predicted Train Set 
    train_plot.scatter(Xtrain[:,:].transpose()[1:,:].tolist()[0], 
                       YTrain_Hat_Plot,
                       color ='Green',
                       label = 'Predicted Train Set')
    train_plot.legend()    

    # Test set  Plotted
    test_plot.grid(True)
    test_plot.set_title('Test Set')

    test_plot.scatter(Xtest[:,:].transpose()[1:,:].tolist()[0], 
                      Ytest[:,:].transpose()[1:,:].tolist()[0],
                      color ='Red',
                      label = 'Actual Test Set')
    test_plot.scatter(Xtest[:,:].transpose()[1:,:].tolist()[0], 
                      YTest_Hat_Plot,
                      color = 'Blue',
                      label = 'Predicted Test Set') 
    test_plot.legend()          

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()


    return x