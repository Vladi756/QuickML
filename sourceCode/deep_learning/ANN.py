import pandas as pd
import numpy as np
import sklearn as sk 

import matplotlib as mpl

mpl.use("Agg")


import random
import matplotlib.pyplot as plt


import keras as kr       # the frontend of DNN 
import tensorflow as tf  # the backend of DNN
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Artficial_Neural_Network
def Artficial_Neural_Network(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    '''
    Takes pre processed data and the dataSet which expects the algorithm
    to be placed on its data. Creates an artifical neural network and trains 
    it using the provided data. Outputs accuracy of ANN. 
    '''

    # Formatting the dataSets for analysis
    XTrain = np.array(Xtrain[:,:].transpose()[1:,:].tolist()[0])
    XTest = np.array(Xtest[:,:].transpose()[1:,:].tolist()[0])
    YTrain = np.array(Ytrain[:,:].transpose()[1:,:].tolist()[0])
    YTest = np.array(Ytest[:,:].transpose()[1:,:].tolist()[0])

    ANN_classifier = Sequential() # initialize a ANN object
    ANN_classifier.add(Dense(input_dim = 11 ,      # No. of input neuron (only for 1st layer)
                         units = 6,       # No. of neuron on this layer = (11+1)/2
                         kernel_initializer = 'uniform' ,    # uniform distribution
                         activation = 'relu' ) # activation func
                  )

    # Adding another hidden layer
    ANN_classifier.add(Dense(units = 6,       
                         kernel_initializer = 'uniform',
                         activation = 'relu')
                  )

    # Dependent variable is binary 
    ANN_classifier.add(Dense(units = 1,       
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid')
                  ) # sigmoid fn yeilds probability 

    # Using the compile function to build the ANN datastructure
    ANN_classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    # Training the ANN 
    hist_1 = ANN_classifier.fit(XTrain, YTrain, 
                   batch_size=32,
                   epochs=50, 
                   verbose=False)

   
    hist_2 = ANN_classifier.fit(XTrain, YTrain,
                   batch_size=32,
                   epochs=50, 
                   callbacks=[EarlyStopping(monitor='loss', patience=1)],
                   verbose=False)

    # Visualizing Data With MatPlotLib

    plt.figure(figsize=(10,8))

    ax_acc = plt.subplot('121')
    ax_los = plt.subplot('122')

    ax_los.plot(hist_1.history['loss'],'r*:',label='loss w/o ES');
    ax_los.plot(hist_2.history['loss'],'g*:',label ='loss with ES');
    plt.plot([0,len(hist_1.history['loss']),],
         [hist_1.history['loss'][-1],hist_1.history['loss'][-1]],'r:')
    plt.plot([0,len(hist_1.history['loss'])],
         [hist_2.history['loss'][-1],hist_2.history['loss'][-1]],'g:')

    ax_los.set_xlabel('Iteration')
    ax_los.set_ylabel('Loss')
    ax_los.legend()

    ax_acc.plot(hist_1.history['accuracy'],'r+:',label='accuracy w/o ES');
    ax_acc.plot(hist_2.history['accuracy'],'g+:',label ='accuracy with ES');
    ax_acc.plot([0,len(hist_1.history['accuracy'])],
         [hist_1.history['accuracy'][-1],hist_1.history['accuracy'][-1]],'r:')
    ax_acc.plot([0,len(hist_1.history['accuracy'])],
         [hist_2.history['accuracy'][-1],hist_2.history['accuracy'][-1]], 'g:')

    ax_acc.legend()
    ax_acc.set_xlabel('Iteration')
    ax_acc.set_ylabel('Accuracy')

    plt.title(f'Artifical Neural Network for {dataSet}')

    plt.legend() 
    
    filename = f'{random.randint(100,999)}'
    plt.savefig(f'../QuickML/webapp/static/{filename}.jpg')

    x = f'../QuickML/webapp/static/{filename}.jpg'

    # clears the mat plot lib cache so other figures can be 
    # created and saved 
    plt.clf()

    return x