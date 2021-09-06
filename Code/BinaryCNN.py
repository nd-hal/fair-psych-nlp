
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.metrics import *
from tensorflow.keras.models import Model
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


def buildAndTrainModel(model, learningRate, batchSize, epochs, trainingData, validationData, testingData, trainingLabels, validationLabels, testingLabels, MODEL_NAME, isPrintModel=True):
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
        
        if valAUC >= max(ValAuc):
            max_predictionsProbs = predictionsProbs
            max_predictions = predictions
        
            
    return Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, ValAuc, max_predictionsProbs, max_predictions


            


# In[35]:


def attachOutputLayerToModel(lastDenseLayer, modelInputs):
    """Take as input a dense layer and attach an output layer"""
    output = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(lastDenseLayer)
    model = Model(inputs=modelInputs, outputs=output)
    return model


# In[30]:


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

def concatenateModels(modelDenseLayers):
    """Get a list if dense layers and concatenate them together"""
    concatenatedModel = layers.concatenate(modelDenseLayers)
    concatenatedModel = layers.Dense(64, activation = 'relu')(concatenatedModel)
    """You can add more layers here after the concatenation of models e.g you can add single/multiple dense layers after the concatenation layer to give the model more power"""
    return concatenatedModel


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
        Accuracy, AUCScore, precisionLow, precisionHigh, recallLow, recallHigh, f1_scoreLow, f1_scoreHigh, Epoch, batch_size, VALAccuracy, VALAUC, max_predictionsProbs, max_predictions = buildAndTrainModel(wordCNNModel, LEARNING_RATE, BATCH_SIZE, EPOCHS, [trainFeatures,trainFeatures,trainFeatures], [validationFeatures,validationFeatures,validationFeatures], [testFeatures,testFeatures,testFeatures], ytrain, yval, ytest, "WordCNNWithoutPretrainedEmbeddings")
        
        Dataset = []
        
        Dataset.append(filename)
        
    
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
        # results_string = 'Results/WordCNN/results//{}_{}results.csv'.format(i, filename)
        # print(results)
        # results_df.to_csv(results_string, index = False)
        
        # predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
        # predictions_df.to_csv('Results/WordCNN/predictions/{}_{}_Binarypredictions.csv'.format(i, filename), index=False)
           
        