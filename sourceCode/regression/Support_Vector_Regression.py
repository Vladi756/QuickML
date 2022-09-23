import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression    
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler    
    
# SUPPORT VECTOR REGRESSION 
def supportVectorRegression(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    
    mpl.rcParams['figure.figsize'] = [11, 6]

    # Formatting the dataSets for analysis
    XTrain = Xtrain[:,:].transpose()[1:,:].tolist()[0]
    XTest = Xtest[:,:].transpose()[1:,:].tolist()[0]
    YTrain = Ytrain[:,:].transpose()[1:,:].tolist()[0]
    YTest = Ytest[:,:].transpose()[1:,:].tolist()[0]

    # SVR doesn't need a train test split, so the individual components 
    # are combined to form the original dataset (except now its pre-processed)
    X_combined = np.r_[XTrain, XTest]
    Y_combined = np.r_[YTrain, YTest]

    #==========================================================#

    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    X_combined = sc_X.fit_transform(X_combined.reshape(-1,1))
    Y_combined = sc_Y.fit_transform(Y_combined.reshape(-1,1))

    # Creating an SVR regressor
    svr_reg = SVR(kernel='poly')
    svr_reg.fit(X_combined, Y_combined)


    plt.title(f'Support Vector Regression for {dataSet}')

    # Scattering actual results
    plt.scatter(X_combined, Y_combined, color = 'blue', label ='Actual')

    X_combined_Plot = X_combined.tolist()

    # Plotting predicted values via linear regression
    plt.plot(sorted(X_combined_Plot), 
             sorted(svr_reg.predict(X_combined)), 
             color = 'orange', 
             label = 'Support Vector')


    plt.legend()          

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x