
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import gensim
from sklearn.metrics import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import *
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[5]:


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
        #model.save('Results/StructuredBinary/{}/epoch_{}'.format(filename,epoch))
            
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
        
        print('Epoch: {}, AUC: {}, F1_High: {}'.format(epoch, aucScore, f1_score2))
        
        if valAUC >= max(ValAuc):
            max_predictionsProbs = predictionsProbs
            max_predictions = predictions
            
    return Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, ValAuc, max_predictionsProbs, max_predictions
            

            


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
    #wordInput = layers.Input(shape=trainFeatures.shape[1:], dtype='float32')
    wordLSTM = wordInput = layers.Input(shape=trainFeatures.shape[1:], dtype='float32', name = 'wordInput')
 
    ## word convolutional neural network
    if isPreTrainedEmbeddings == False:
        # Create embeddings using keras built in function.
        wordLSTM = layers.Embedding(input_dim=vocabularyWords + 1, 
                                   output_dim=embeddingsDimensionality, 
                                   input_length=len(trainFeatures[0]))(wordInput)
    
        # Add CNN layers
        for i in range(numLstmLayers):

            #name = 'layer_bidirectional_lstm_{0}'.format(i+1)

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

            #name = 'layer_bidirectional_lstm_{0}'.format(i+1)

            if( (i==0) and (numLstmLayers==1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2,
                    return_sequences=False))(wordLSTM)
            elif(i != (numLstmLayers-1) ):
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=True))(wordLSTM)
            else:
                wordLSTM = layers.Bidirectional(layers.LSTM(units=lstmNodes, dropout=0.2, recurrent_dropout=0.2, 
                    return_sequences=False))(wordLSTM)
    
    #wordLSTM = layers.Dropout(0.5)(wordLSTM)
    denseLayer = layers.Dense(128)(wordLSTM)
    
    
    return denseLayer, wordInput


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
        
        xtrain = data_train[1]
        ytrain = data_train[0]
        ytrain = ytrain.replace(binary_mapper)
    
        xtest = data_test[1]
        ytest = data_test[0]
        ytest = ytest.replace(binary_mapper)
    
        xval = data_val[1]
        yval = data_val[0]
        yval = yval.replace(binary_mapper)
        
        
        # Create input feature arrays
    ##################################################### You can set the embedding path to REPRESENTATION EMBEDDINGS too which you can find in "RepresentationEmbeddings" folder ################################
        EMBEDDING_PATH = "glove.840B.300d.txt"
        
        VocabSize = None
        trainFeatures, validationFeatures, testFeatures, WORDS_TO_KEEP = loadDataForWordsWithPreTrainedEmbeddings(xtrain,xval,xtest, EMBEDDING_PATH)
    
    # Build WordCNN model
        EMBEDDINGS_DIMENSIONALITY = 300 # don't need this now
        LSTM_NODES_IN_LAYERS = 64
        NUM_LSTM_LAYERS = 3
        wordLSTMDenseLayer, wordLSTMInput = createWordLSTM(trainFeatures, validationFeatures, testFeatures, 
                                                           NUM_LSTM_LAYERS, WORDS_TO_KEEP, EMBEDDINGS_DIMENSIONALITY, LSTM_NODES_IN_LAYERS, bidirectional=True, isPreTrainedEmbeddings=True)
    
    # Attach the output layer with the model
        wordLSTMModel = attachOutputLayerToModel(wordLSTMDenseLayer, wordLSTMInput)
    
    # Train model
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 32
        EPOCHS = 50
        Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, VALAUC, max_predictionsProbs, max_predictions = buildAndTrainModel(wordLSTMModel, LEARNING_RATE, BATCH_SIZE, EPOCHS, trainFeatures, validationFeatures, testFeatures, ytrain, ytest, yval, "WordCNNWithoutPretrainedEmbeddings")
        
    
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
                'ValAccuracy':VALAccuracy,
                'ValAuc': VALAUC
                }
        
        predictions_dictionary = {
                    'label': max_predictions,
                    'sentence': xtest,
                    'probs': max_predictionsProbs
                    }
                
        # results_df = pd.DataFrame.from_dict(results)
        # results_string = 'Results/WordLSTM/results/{}_{}results.csv'.format(i, filename)
        # results_df.to_csv(results_string, index = False)
        
        # predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        # predictions_df.to_csv('Results/WordLSTM/predictions/{}_{}_Binarypredictions.csv'.format(i, filename), index=False)
    