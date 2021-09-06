

import tensorflow as tf
import random
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import gensim
import scipy.stats as st
from sklearn.metrics import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import sklearn
from tensorflow.keras.preprocessing import *
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[5]:

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

# In[18]:


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


# In[63]:


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


            


# In[35]:


def attachOutputLayerToModel(lastDenseLayer, modelInputs):
    """Take as input a dense layer and attach an output layer"""
    output = layers.Dense(1, activation='sigmoid')(lastDenseLayer)
    model = Model(inputs=modelInputs, outputs=output)
    return model


# In[30]:


def createWordLSTM(trainFeatures, validationFeatures, testFeatures, numLstmLayers, vocabularyWords, embeddingsDimensionality, lstmNodes, bidirectional, isPreTrainedEmbeddings):  
    """Create a word lstm"""
    
    ## create basic cnn model
    wordLSTM = wordInput = layers.Input(shape=trainFeatures.shape[1:], dtype='float32', name = 'wordInput')
 
    ## word convolutional neural network
    if isPreTrainedEmbeddings == False:
        # Create embeddings using keras built in function.
        wordLSTM = layers.Embedding(input_dim=vocabularyWords + 1, 
                                   output_dim=embeddingsDimensionality, 
                                   input_length=len(trainFeatures[0]))(wordInput)
    
        # Add CNN layers
        for i in range(numLstmLayers):


            if( (i==0) and (numLstmLayers==1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=False))(wordLSTM)
            elif(i != (numLstmLayers-1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=True))(wordLSTM)
            else:
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=False))(wordLSTM)
    else:
        
        for i in range(numLstmLayers):


            if( (i==0) and (numLstmLayers==1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2,
                    return_sequences=False))(wordLSTM)
            elif(i != (numLstmLayers-1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=True))(wordLSTM)
            else:
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=False))(wordLSTM)
    
    denseLayer = layers.Dense(64)(wordLSTM)
    
    
    return denseLayer, wordInput


# In[31]:


def loadDataForWordsWithPreTrainedEmbeddings(trainText, validationText, testText, EMBEDDING_PATH):
    """This function takes as input three text files and and a pre-trained word embedding file and returns arrays containing word embeddings for each word in the text. These arrays can be used 
    directly in a keras model without the use of keras.layers.Embedding layer."""
    
    # Load embeddings
    EMBEDDING_SIZE = 300
    embeddingsDataDictionary = {}
    
    with open(EMBEDDING_PATH, 'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddingsDataDictionary[word] = vector
    
    ## convert words into word ids
    meanLength = np.mean([len(item.split(" ")) for item in trainText])
    MAX_SENTENCE_LENGTH = int(meanLength + 100) # we let a sentence go 100 words longer than the mean sentence length.
    
    ## convert train, validation, and test text into lists with word ids
    trainTextWords = [splitRowIntoWords(row, MAX_SENTENCE_LENGTH) for row in trainText]
    trainWordFeatures = []
    for row in trainTextWords:
        rowEmbeddings = [embeddingsDataDictionary[word] if word in embeddingsDataDictionary else [0]*EMBEDDING_SIZE for word in row]
        trainWordFeatures.append(rowEmbeddings)

    validationTextWords = [splitRowIntoWords(row, MAX_SENTENCE_LENGTH) for row in validationText]
    validationWordFeatures = []
    for row in validationTextWords:
        rowEmbeddings = [embeddingsDataDictionary[word] if word in embeddingsDataDictionary else [0]*EMBEDDING_SIZE for word in row]
        validationWordFeatures.append(rowEmbeddings)

    testTextWords = [splitRowIntoWords(row, MAX_SENTENCE_LENGTH) for row in testText]
    testWordFeatures = []
    for row in testTextWords:
        rowEmbeddings = [embeddingsDataDictionary[word] if word in embeddingsDataDictionary else [0]*EMBEDDING_SIZE for word in row]
        testWordFeatures.append(rowEmbeddings)
    
    return np.array(trainWordFeatures), np.array(validationWordFeatures), np.array(testWordFeatures), None

def loadDataForWordsWithoutPreTrainedEmbeddings(trainText, validationText, testText, VOCABULARY_SIZE):
    """This function takes as input three text files and a vocabulary size (words to keep in the model) and returns lists containing word ids and the total number of words for the model.
    The word ids will be used the keras.layers.Embedding function to create word embeddings"""
    
    # If vocabulary size is not defined, use all words
    if VOCABULARY_SIZE == None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(trainText)
        WORDS_TO_KEEP = len(tokenizer.word_index)
        
    # Delete words based on frequency and keep words equal to the vocabulary size
    else:
        tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
        tokenizer.fit_on_texts(trainText)
        WORDS_TO_KEEP = VOCABULARY_SIZE
    
    ## convert words into word ids
    meanLength = np.mean([len(item.split(" ")) for item in trainText])
    MAX_SENTENCE_LENGTH = int(meanLength + 100) # we let a sentence go 100 words longer than the mean sentence length.
    
    ## convert train, validation, and test text into lists with word ids
    trainWordFeatures = tokenizer.texts_to_sequences(trainText)
    trainWordFeatures = pad_sequences(trainWordFeatures, MAX_SENTENCE_LENGTH, padding='post')

    validationWordFeatures = tokenizer.texts_to_sequences(validationText)
    validationWordFeatures = pad_sequences(validationWordFeatures, MAX_SENTENCE_LENGTH, padding='post')

    testWordFeatures = tokenizer.texts_to_sequences(testText)
    testWordFeatures = pad_sequences(testWordFeatures, MAX_SENTENCE_LENGTH, padding='post')
    
    return trainWordFeatures, validationWordFeatures, testWordFeatures, WORDS_TO_KEEP


# In[32]:

files = ['TrustPhys_','SubjectiveLit_','Anxiety_','Numeracy_']


cv = ['1','2','3','4','5']

# In[69]:


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
    
        binary_mapper = {-1: 0}
        
        xtrain = data_train[1]
        ytrain = data_train[0]
    
        xtest = data_test[1]
        ytest = data_test[0]
    
        xval = data_val[1]
        yval = data_val[0]
        
        
        # Create input feature arrays
    ##################################################### You can set the embedding path to REPRESENTATION EMBEDDINGS too which you can find in "RepresentationEmbeddings" folder ################################
        EMBEDDING_PATH = "glove.840B.300d.txt"
        
        VocabSize = None
        trainFeatures, validationFeatures, testFeatures, WORDS_TO_KEEP = loadDataForWordsWithPreTrainedEmbeddings(xtrain,xval,xtest, EMBEDDING_PATH)
    
    # Build WordCNN model
        EMBEDDINGS_DIMENSIONALITY = 300 # don't need this now
        LSTM_NODES_IN_LAYERS = 128
        NUM_LSTM_LAYERS = 2
        wordLSTMDenseLayer, wordLSTMInput = createWordLSTM(trainFeatures, validationFeatures, testFeatures, 
                                                           NUM_LSTM_LAYERS, WORDS_TO_KEEP, EMBEDDINGS_DIMENSIONALITY, LSTM_NODES_IN_LAYERS, bidirectional=True, isPreTrainedEmbeddings=True)
    
    # Attach the output layer with the model
        wordLSTMModel = attachOutputLayerToModel(wordLSTMDenseLayer, wordLSTMInput)
    
    # Train model
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 16
        EPOCHS = 25
        MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, ValMSE, PearsonR, Epochs, pred = buildAndTrainModel(wordLSTMModel, LEARNING_RATE, BATCH_SIZE, EPOCHS, trainFeatures, validationFeatures, testFeatures, ytrain, yval, ytest, "WordCNNWithoutPretrainedEmbeddings")
        
    
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
                    'sentence': np.array(xtest).flatten(),
                    'pred': np.array(pred).flatten()
                    }
    
        #results_df = pd.DataFrame.from_dict(results)
        #results_string = 'Results/WordLSTM/{}_{}Conresults.csv'.format(i, filename)
        #results_df.to_csv(results_string, index = False)
        
        #predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        #predictions_df.to_csv('Results/WordLSTM/{}_{}_Conpredictions.csv'.format(i, filename), index=False)
           