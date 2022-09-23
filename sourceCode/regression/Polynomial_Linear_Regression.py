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


# POLYNOMIAL LINEAR REGRESSION 
def polynomialLinearRegression(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    
    # Formatting the dataSets for analysis
    XTrain = Xtrain[:,:].transpose()[1:,:].tolist()[0]
    XTest = Xtest[:,:].transpose()[1:,:].tolist()[0]
    YTrain = Ytrain[:,:].transpose()[1:,:].tolist()[0]
    YTest = Ytest[:,:].transpose()[1:,:].tolist()[0]

    # PLR doesn't need a train test split, so the individual components 
    # are combined to form the original dataset (except now its pre-processed)
    X_combined = np.r_[XTrain, XTest]
    Y_combined = np.r_[YTrain, YTest]

    # Manually casting to int 
    X_combined = np.array(X_combined, dtype='int')
    Y_combined = np.array(Y_combined, dtype='int')

    lin_reg = LinearRegression()
    lin_reg.fit(X_combined.reshape(-1,1), Y_combined)

    # Creating a polynomial regressor
    poly_reg = PolynomialFeatures(degree=3)
    # Transforming X from just X to X + its polynomial terms
    X_Comb_Poly = X_combined.reshape(-1,1)
    X_poly = poly_reg.fit_transform(X_Comb_Poly)

    # New linear regression fitted onaugmented X matrix and 
    # original Y vector. 
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X_poly, Y_combined)

    plt.title(f'Polynomial Linear Regression for {dataSet}')

    # Scattering actual results
    plt.scatter(X_combined, Y_combined, color = 'red', label ='Actual')

    # Plotting predicted values via linear regression
    plt.plot(X_combined, 
             lin_reg.predict(X_combined.reshape(-1,1)), 
             color = 'blue', 
             label = 'Linear')

    print(f'{X_poly}============')

    X_combined_Plot = X_combined.tolist()
    X_poly_Plot = X_poly.tolist()

    # Plotting predicted values via polynomial regression
    plt.plot(sorted(X_combined_Plot),
             lin_reg2.predict(sorted(X_poly_Plot)), 
             # lin_reg2.predict(poly_reg.fit_transform(X_combined.reshape(-1,1))), 
             color = 'green', 
             label = 'Poylnomial')


    plt.legend()          

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x