


# Import packages
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.metrics import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import *
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# In[ ]:


def getMetrics(trueLabels, predictedLabels, predictedLabelsProbabilities):
    """Takes as input true labels, predictions, and prediction confidence scores and computes all metrics"""
    accuracy = accuracy_score(trueLabels,predictedLabels)
    aucScore = round(roc_auc_score(trueLabels, predictedLabelsProbabilities) * 100, 1)
    precisionLow = round(precision_score(trueLabels, predictedLabels, average=None)[0] * 100, 1)
    precisionHigh = round(precision_score(trueLabels, predictedLabels, average=None)[1] * 100, 1)
    recallLow = round(recall_score(trueLabels, predictedLabels, average=None)[0] * 100, 1)
    recallHigh = round(recall_score(trueLabels, predictedLabels, average=None)[1] * 100, 1)
    fpr, tpr, threshold = roc_curve(trueLabels, predictedLabelsProbabilities)
    f1_scoreLow = round(f1_score(trueLabels,predictedLabels, average=None)[0]*100,1)
    f1_scoreHigh = round(f1_score(trueLabels,predictedLabels, average=None)[1]*100,1)
    
    return accuracy, aucScore, precisionLow, precisionHigh, recallLow, recallHigh, fpr, tpr, f1_scoreLow, f1_scoreHigh


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


def buildAndTrainModel(model, learningRate, batchSize, epochs, trainingData, validationData, testingData, trainingLabels, testingLabels, validationLabels, MODEL_NAME, isPrintModel=True):
    """Take the model and model parameters, build and train the model"""
    
    # Build and compile model
    # To use other optimizers, refer to: https://keras.io/optimizers/
    # Please do not change the loss function
    
    optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    if isPrintModel:
        print(model.summary())
        

    
    for epoch in range(0, epochs):
        model.fit(trainingData, trainingLabels,
                        epochs=1,
                        verbose=0,
                        batch_size=batchSize,
                        shuffle=False)
        
        # Evaluate model
        trainLoss, trainAccuracy, trainAUC = model.evaluate(trainingData, trainingLabels, verbose=False)
        valLoss, valAccuracy, valAUC = model.evaluate(validationData, validationLabels, verbose=False)
            
            ## get metrics
        predictionsProbs = model.predict(testingData)
        predictions = [1 if value >= 0.5 else 0 for value in predictionsProbs]
        predictionsProbs = [value for value in predictionsProbs]
        accuracy, aucScore, precision1, precision2, recall1, recall2, fpr, tpr, f1_score1, f1_score2 = getMetrics(testingLabels, predictions, predictionsProbs)
        Accuracy.append(accuracy)
        AUCScore.append(aucScore)
        precisionLow.append(precision1)
        precisionHigh.append(precision2)
        recallLow.append(recall1)
        recallHigh.append(recall2)
        f1_scoreLow.append(f1_score1)
        f1_scoreHigh.append(f1_score2)
        Epoch.append(epoch)
        batch_size.append(batchSize)
        VALAccuracy.append(valAccuracy)
        ValAuc.append(valAUC)
        
        if valAUC >= max(ValAuc):
            max_predictionsProbs = predictionsProbs
            max_predictions = predictions
            
    return Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, ValAuc, max_predictionsProbs, max_predictions


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
    
        Accuracy = []
        AUCScore = []
        precisionHigh = []
        precisionLow = []
        recallHigh = []
        recallLow = []
        f1_scoreLow = []
        f1_scoreHigh = []
        Epoch = []
        batch_size = []
        VALAccuracy = []
        ValAuc = []
    
        string_train = 'MedianCV/{}/{}train.txt'.format(i, filename)
        string_test = 'MedianCV/{}/{}test.txt'.format(i, filename)
        string_val = 'MedianCV/{}/{}val.txt'.format(i, filename)
        data_train = pd.read_csv(string_train, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
        data_test = pd.read_csv(string_test, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
        data_val = pd.read_csv(string_val, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    
        binary_mapper = {-1: 0}
        vectorizer = CountVectorizer(ngram_range = (1,1), max_features = 50000, binary = True)
        
        xtrain = data_train[1]
        x_train = data_train[1]
        xtrainfeatures = vectorizer.fit_transform(x_train).toarray()
        ytrain = data_train[0]
        ytrain = ytrain.replace(binary_mapper)
    
        xtest = data_test[1]
        x_test = data_test[1]
        xtestfeatures = vectorizer.transform(x_test).toarray()
        ytest = data_test[0]
        ytest = ytest.replace(binary_mapper)
    
        xval = data_val[1]
        x_val = data_val[1]
        xvalfeatures = vectorizer.transform(x_val).toarray()
        yval = data_val[0]
        yval = yval.replace(binary_mapper)
    
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
        Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, VALAUC, max_predictionsProbs, max_predictions = buildAndTrainModel(NNModel, LEARNING_RATE, 
                                                                                                                                              BATCH_SIZE, EPOCHS, 
                                                                                                                                                  xtrainfeatures,xvalfeatures,
                                                                                                                                                  xtestfeatures, ytrain,ytest,yval,"NN")
    
        results = {
                'AUCScore': AUCScore,
                'Accuracy': Accuracy,
                'PrecisionHigh': precisionHigh,
                'PrecisionLow': precisionLow,
                'RecallHigh': recallHigh,
                'RecallLow': recallLow,
                'F1_ScoreLow': f1_scoreLow,
                'F1_ScoreHigh': f1_scoreHigh,
                'Epoch': Epoch,
                'BatchSize': batch_size,
                'ValAccuracy': VALAccuracy,
                'ValAuc': VALAUC
                }
        
        predictions_dictionary = {
                    'label': max_predictions,
                    'sentence': xtest,
                    'probs': max_predictionsProbs
                    }
    
        #results_df = pd.DataFrame.from_dict(results)
        #results_string = 'Results/StructuredBinary/{}_{}Binaryresults.csv'.format(i, filename)
        #results_df.to_csv(results_string, index = False)
        
        #predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        #predictions_df.to_csv('Results/StructuredBinary/{}_{}_Binarypredictions.csv'.format(i, filename), index=False)


# In[ ]:




