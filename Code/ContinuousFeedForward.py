
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats as st
from tensorflow.keras import layers
import sklearn.metrics
import sklearn
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import *
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
# In[ ]:
#https://github.com/wwbp/empathic_reactions/blob/master/modeling/main/crossvalidation/experiment.py
def correlation(true, pred):
    pred = np.array(pred).flatten()
    result = st.pearsonr(np.array(true),pred)
    return result[0]

def getMetrics(trueLabels, predictedLabels):
    """Takes as input true labels, predictions, and prediction confidence scores and computes all metrics"""
    MSE = sklearn.metrics.mean_squared_error(trueLabels, predictedLabels, squared = True)
    MAE = sklearn.metrics.mean_absolute_error(trueLabels, predictedLabels)
    MAPE = sklearn.metrics.mean_absolute_percentage_error(trueLabels, predictedLabels)
    RMSE = sklearn.metrics.mean_squared_error(trueLabels, predictedLabels, squared = False)
    PearsonR = correlation(true = trueLabels,
                           pred = predictedLabels)
    
    return MSE, MAE, MAPE, RMSE, PearsonR


# In[ ]:


def splitRowIntoWords(row, length):
    """Takes a variable length text input and convert it into a list of words with length equal to 'length' in the function parameter"""
    words = tf.keras.preprocessing.text.text_to_word_sequence(row, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\'', lower=True, split=" ")
    
    # If length is less than required length, add zeros
    while len(words) < length:
        words.append(0)
        
    # If greater, remove stuff at the end
    if len(words) >= length:
        words = words[:length]
        
    return words


# In[ ]:


def buildAndTrainModel(model, learningRate, batchSize, epochs, trainingData, validationData, testingData, trainingLabels, validationLabels, testingLabels, MODEL_NAME, isPrintModel=True):
    """Take the model and model parameters, build and train the model"""
    
    # Build and compile model
    # To use other optimizers, refer to: https://keras.io/optimizers/
    # Please do not change the loss function
    
    optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError())
    
    if isPrintModel:
        print(model.summary())
        

    
    for epoch in range(0, epochs):
        model.fit(trainingData, trainingLabels,
                        epochs=1,
                        verbose=0,
                        batch_size=batchSize,
                        shuffle=False)
        
        # Evaluate model
        valLoss = model.evaluate(validationData, validationLabels, verbose=False)
        #model.save('Results/StructuredBinary/{}/epoch_{}'.format(filename,epoch))
            
            ## get metrics
        predictions = model.predict(testingData)
        MSE, MAE, MAPE, RMSE, PR = getMetrics(testingLabels,predictions)
        
        MeanSquaredError.append(MSE)
        RootMeanSquaredError.append(RMSE)
        MeanAbsoluteError.append(MAE)
        MeanAbsolutePercentageError.append(MAPE)
        PearsonR.append(PR)
        ValMSE.append(valLoss)
        Epoch.append(epoch)
        
        if valLoss <= min(ValMSE):
            max_predictions = predictions
       
        
            
    return MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, ValMSE, PearsonR, Epoch, max_predictions


# In[ ]:


def attachOutputLayerToModel(lastDenseLayer, modelInputs):
    """Take as input a dense layer and attach an output layer"""
    output = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(lastDenseLayer)
    model = Model(inputs=modelInputs, outputs=output)
    return model


# In[ ]:


def createFeedForwardNeuralNetwork(trainFeatures, validationFeatures, testFeatures, numLayers, layerNodes): 
    """Create a feed forward neural network"""
    
    ## create basic nn model
    modelInput = layers.Input(shape=trainFeatures.shape[1:], dtype='float32')
    neuralNetworkLayer = layers.Dense(layerNodes, activation='relu', input_shape=trainFeatures.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(0.001))(modelInput)
    neuralNetworkLayer = layers.Dropout(0.5)(neuralNetworkLayer)
    for i in range(numLayers - 1):
        neuralNetworkLayer = layers.Dense(layerNodes, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(neuralNetworkLayer)
        neuralNetworkLayer = layers.Dropout(0.5)(neuralNetworkLayer)
    
    # You can change the number of nodes in the dense layer. Right now, it's set to 32.
    denseLayer = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(neuralNetworkLayer)
    
    return denseLayer, modelInput


# In[ ]:



files = ['TrustPhys_','SubjectiveLit_','Anxiety_','Numeracy_']


cv = ['1','2','3','4','5']


# In[ ]:


for filename in files:
    
    for i in cv:
    
        MeanSquaredError = []
        MeanAbsoluteError = []
        MeanAbsolutePercentageError = []
        RootMeanSquaredError = []
        PearsonR = []
        Epoch = []
        ValMSE = []
    
        string_train = 'ContinuousCV/{}/{}train.txt'.format(i, filename)
        string_test = 'ContinuousCV/{}/{}test.txt'.format(i, filename)
        string_val = 'ContinuousCV/{}/{}val.txt'.format(i, filename)
        data_train = pd.read_csv(string_train, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
        data_test = pd.read_csv(string_test, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
        data_val = pd.read_csv(string_val, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    
        vectorizer = CountVectorizer(ngram_range = (1,1), max_features = 50000, binary = True)
        
        x_train = data_train[1]
        xtrainfeatures = vectorizer.fit_transform(x_train).toarray()
        ytrain = data_train[0]
    
        x_test = data_test[1]
        xtestfeatures = vectorizer.transform(x_test).toarray()
        ytest = data_test[0]
    
        x_val = data_val[1]
        xvalfeatures = vectorizer.transform(x_val).toarray()
        yval = data_val[0]
    
    #Build StructuredNN
        StructuredLayers = 3
        StructuredUnits = 256
        
        StructuredDenseLayer, StructuredInput = createFeedForwardNeuralNetwork(xtrainfeatures,xvalfeatures,xtestfeatures,StructuredLayers,StructuredUnits)
    
        
        # Attach the output layer with the model
        NNModel = attachOutputLayerToModel(StructuredDenseLayer, StructuredInput)
    
    # Train model
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 32
        EPOCHS = 35
        MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, ValMSE, PearsonR, Epochs, pred = buildAndTrainModel(NNModel, LEARNING_RATE, 
                                                                                                                                              BATCH_SIZE, EPOCHS, 
                                                                                                                                                  xtrainfeatures,xvalfeatures,xtestfeatures, ytrain,yval,ytest,"NN")
    
        results = {
                    'Epochs': Epochs,
                    'Mean_Squared_Error': MeanSquaredError,
                    'Root_Mean_Squared_Error': RootMeanSquaredError,
                    'Mean_Absolute_Error': MeanAbsoluteError,
                    'Mean_Absolute_Percentage_Error': MeanAbsolutePercentageError,
                    'PearsonR': PearsonR,
                    'Val_Mean_Squared_Error': ValMSE
                    }
        
        predictions_dictionary = {
                    'sentence': np.array(x_test).flatten(),
                    'pred': np.array(pred).flatten()
                    }
    
        #results_df = pd.DataFrame.from_dict(results)
        #results_string = 'Results/StructuredContinue/{}_{}Conresults.csv'.format(i, filename)
        #results_df.to_csv(results_string, index = False)
        
        #predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        #predictions_df.to_csv('Results/StructuredContinue/{}_{}_Conpredictions.csv'.format(i, filename), index=False)
            

