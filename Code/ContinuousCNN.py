

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
        trainLoss = model.evaluate(trainingData, trainingLabels, verbose=False)
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


def createWordCNN(trainFeatures, validationFeatures, testFeatures, numConvLayers, vocabularyWords, embeddingsDimensionality, numFilters, kernel, isPreTrainedEmbeddings): 
    """Create a word cnn"""
    
    ## create basic cnn model
    wordInput = layers.Input(shape=trainFeatures.shape[1:], dtype='float32')
 
    ## word convolutional neural network
    if isPreTrainedEmbeddings == False:
        # Create embeddings using keras built in function.
        wordCNN = layers.Embedding(input_dim=vocabularyWords + 1, 
                                   output_dim=embeddingsDimensionality, 
                                   input_length=len(trainFeatures[0]))(wordInput)
    
        # Add CNN layers equal to numConvLayers
        for i in range(numConvLayers):
            wordCNN = layers.Conv1D(numFilters, kernel, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(wordCNN)
            wordCNN = layers.Dropout(0.5)(wordCNN)
    else:
        
        # Here, we are using pre-trained embeddings. Therefore, we don't need to call layers.embeddings function.
        wordCNN = layers.Conv1D(numFilters, kernel, activation='relu', input_shape=trainFeatures.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(0.001))(wordInput)
        wordCNN = layers.Dropout(0.5)(wordCNN)
        for i in range(numConvLayers - 1):
            wordCNN = layers.Conv1D(numFilters, kernel, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(wordCNN)
            wordCNN = layers.Dropout(0.5)(wordCNN)
    
    # GlobalMaxPooling is a good function to use for pooling operations, let's keep it like this
    wordCNN = layers.GlobalMaxPooling1D()(wordCNN)
    wordCNN = layers.Dropout(0.5)(wordCNN)
    
    # You can change the number of nodes in the dense layer. Right now, it's set to 64.
    denseLayer = layers.Dense(64)(wordCNN)
    
    return denseLayer, wordInput  

def concatenateModels(modelDenseLayers):
    """Get a list if dense layers and concatenate them together"""
    concatenatedModel = layers.concatenate(modelDenseLayers)
    concatenatedModel = layers.Dense(64, activation = 'relu')(concatenatedModel)
    """You can add more layers here after the concatenation of models e.g you can add single/multiple dense layers after the concatenation layer to give the model more power"""
    return concatenatedModel         


# In[35]:


def attachOutputLayerToModel(lastDenseLayer, modelInputs):
    """Take as input a dense layer and attach an output layer"""
    output = layers.Dense(1, activation='sigmoid')(lastDenseLayer)
    model = Model(inputs=modelInputs, outputs=output)
    return model



# In[31]:


def loadDataForWordsWithPreTrainedEmbeddings(trainText, validationText, testText, EMBEDDING_PATH):
    """This function takes as input three text files and and a pre-trained word embedding file and returns arrays containing word embeddings for each word in the text. These arrays can be used 
    directly in a keras model without the use of keras.layers.Embedding layer."""
    
    # Load embeddings
    #embeddingsData = pd.DataFrame(pd.read_csv(EMBEDDING_PATH, " ",engine='python')).values.tolist()
    #embeddingsDataDictionary = dict([(item[0], item[1:len(item)-1]) for item in embeddingsData]) # create dictionary of key=word, value=word embedding from the embedding file
    #EMBEDDING_SIZE = int(len(embeddingsDataDictionary[random.choice(list(embeddingsDataDictionary.keys()))]))
    
    EMBEDDING_SIZE = 300
    embeddingsDataDictionary = {}
    
    with open(EMBEDDING_PATH, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddingsDataDictionary[word] = vector
    
    ## convert words into word ids
    meanLength = np.mean([len(item.split(" ")) for item in trainText])
    MAX_SENTENCE_LENGTH = int(meanLength + 10) # we let a sentence go 100 words longer than the mean sentence length.
    
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
        FILTERS_SIZE = 256
        EMBEDDINGS_DIMENSIONALITY = 300 # don't need this now
        KERNEL_SIZE1 = 1
        KERNEL_SIZE2 = 2
        KERNEL_SIZE3 = 3
        NUM_CNN_LAYERS = 1
        
        wordCNNDenseLayer1, wordCNNInput1 = createWordCNN(trainFeatures, validationFeatures, testFeatures, NUM_CNN_LAYERS, WORDS_TO_KEEP, EMBEDDINGS_DIMENSIONALITY, FILTERS_SIZE, KERNEL_SIZE1, isPreTrainedEmbeddings=True)
    
        wordCNNDenseLayer2, wordCNNInput2 = createWordCNN(trainFeatures, validationFeatures, testFeatures, NUM_CNN_LAYERS, WORDS_TO_KEEP, EMBEDDINGS_DIMENSIONALITY, FILTERS_SIZE, KERNEL_SIZE2, isPreTrainedEmbeddings=True)
    
        wordCNNDenseLayer3, wordCNNInput3 = createWordCNN(trainFeatures, validationFeatures, testFeatures, NUM_CNN_LAYERS, WORDS_TO_KEEP, EMBEDDINGS_DIMENSIONALITY, FILTERS_SIZE, KERNEL_SIZE3, isPreTrainedEmbeddings=True)
    
        concatenatedDenseLayer = concatenateModels([wordCNNDenseLayer1,wordCNNDenseLayer2,wordCNNDenseLayer3])
    
    # Attach the output layer with the model
        wordCNNModel = attachOutputLayerToModel(concatenatedDenseLayer, [wordCNNInput1,wordCNNInput2,wordCNNInput3])
    
    # Train model
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 32
        EPOCHS = 50
        MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, ValMSE, PearsonR, Epochs, pred = buildAndTrainModel(wordCNNModel, LEARNING_RATE, BATCH_SIZE, EPOCHS,[trainFeatures,trainFeatures,trainFeatures], [validationFeatures,validationFeatures,validationFeatures], [testFeatures,testFeatures,testFeatures], ytrain, yval, ytest, "WordCNNWithoutPretrainedEmbeddings")
        
    
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
        
        # results_df = pd.DataFrame.from_dict(results)
        # results_string = 'Results/WordCNN/ConResults/{}_{}results.csv'.format(i, filename)
        # results_df.to_csv(results_string, index = False)
        
        # predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        # predictions_df.to_csv('Results/WordCNN/ConPredictions/{}_{}_Conpredictions.csv'.format(i, filename), index=False)
     