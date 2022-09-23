import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
# import dash
# import dash_core_components as DCC 
# import dash_html_components as HTML 


# SIMPLE LINEAR REGRESSION
def simpleLinearRegression(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    """
    Takes the train and test split of the dataset, as well as name
    of the uploaded dataSet. Fits a regressor and plots a simple
    linear regression on the dataset. Saves the figure and returns path
    to saved figure as jpg.
    """
    
    regressor = LinearRegression()
    regressor.fit(Xtrain, Ytrain)   

    plt.title(f'Linear Regression for Dataset: {dataSet}')

    plt.scatter(Xtest[:,:].transpose()[1:,:].tolist()[0], 
                Ytest[:,:].transpose()[1:,:].tolist()[0], 
                color='blue',   
                label='Test Samples')
    
    plt.scatter(Xtrain[:,:].transpose()[1:,:].tolist()[0],
                Ytrain[:,:].transpose()[1:,:].tolist()[0],
                color='red', 
                label = 'Train Samples')
  
    XTest_Plot = Xtest[:,:].transpose()[1:,:].tolist()[0]
    (Ytrain[:,:].transpose()[1:,:].tolist()[0])

    Ytrain_temp = regressor.predict(Xtest) 

    YTrain_Hat_Plot = Ytrain_temp.transpose()[1:,:].tolist()[0]

    plt.plot(sorted(XTest_Plot),
             sorted(YTrain_Hat_Plot),
             label='Regression line')

    plt.legend()
    plt.grid()

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x 

