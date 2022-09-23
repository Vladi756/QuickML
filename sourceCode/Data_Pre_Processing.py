import os
import pandas as pd
import json as j
import numpy as np
import sklearn as sk
from csv import reader
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dataPreProcess(dataSet, varMap):


    # Extractin the filename from the file path
    filename = dataSet.split('/')[-1]  
    # Rreading user's choice from the file
    with open('choice.txt') as f: 
        choice = f.read()

    data = pd.read_csv(
        os.path.join('../QuickML/sourceCode', filename))

    varMap['Missing'] = data.columns[data.isnull().any()].tolist()

    data.drop(varMap['Ignored'], axis=1)

    # Splitting Dependent & Independent Variables
    X = data[varMap['Independent']]  # DataFrames
    y = data[varMap['Dependent']]

    # Removing any missing data
    if len(varMap['Missing']) > 0:
        imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
        imputer = imputer.fit(X[varMap['Missing']])
        X[varMap['Missing']] =imputer.transform(X[varMap['Missing']])
	

    # Encoding Categorical Variables  
    if len(varMap['Categorical']) > 0:
        le = LabelEncoder()
        X[varMap['Categorical']]= pd.DataFrame(le.fit_transform(X[varMap['Categorical']]))
        col_tans = make_column_transformer( 
                         (OneHotEncoder(), 
                         varMap['Categorical']))
        # Xtemp2 = col_tans.fit_transform(X[varMap['Categorical']])
    

    # Splitting Into Train and Test Set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 0)

    if choice != 'ML-CLU-HC' and choice != 'ML-REG-SVfR' and choice != 'ML-REG-SLR' and choice != 'ML-REG-MLR' and choice != 'ML-REG-PLR':
        # Feature Scaling
        scale_X = StandardScaler()
        X_train.iloc[: , :] = scale_X.fit_transform(X_train.iloc[: , :])
        X_test.iloc[: , :] = scale_X.fit_transform(X_test.iloc[: , :])

        scale_Y = StandardScaler()
        y_train.iloc[: , :] = scale_Y.fit_transform(y_train.iloc[: , :])
        y_test.iloc[: , :] = scale_Y.fit_transform(y_test.iloc[: , :])

    # Returns a dictionary of preprocessed data
    ret = {
            'X_train': X_train.values.tolist(),
            'X_test': X_test.values.tolist(),
            'y_train': y_train.values.tolist(),
            'y_test': y_test.values.tolist()
    }

    return(ret)