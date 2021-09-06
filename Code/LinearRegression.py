
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pandas as pd
import numpy as np
from scipy import stats as st
#from sklearn.utils.testing import ignore_warnings
import sklearn.metrics
import matplotlib.pyplot as plt
import shap

#CLFSwitcher:https://stackoverflow.com/questions/51695322/compare-multiple-algorithms-with-sklearn-pipeline
#@ignore_warnings(category=ConvergenceWarning)
class CLFSwitcher(BaseEstimator):
    def __init__(
            self,
            estimator = SGDClassifier(),
            ):
        """
        Allows for the use of gridsearch for all hyperparameter settings
        between preprocessing steps and classifiers
        """
        self.estimator = estimator
        
    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X,y)
        return self
    
    def predict(self, X, y=None):
        return self.estimator.predict(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)
    
def load_data(i, filename):
    string_train = 'ContinuousCV/{}/{}train.txt'.format(i, filename)
    string_test = 'ContinuousCV/{}/{}test.txt'.format(i, filename)
    string_val = 'ContinuousCV/{}/{}val.txt'.format(i, filename)
    data_train = pd.read_csv(string_train, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    data_test = pd.read_csv(string_test, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    data_val = pd.read_csv(string_val, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    
    xtrain = data_train[1]
    xtrain = np.array(xtrain)
    ytrain = data_train[0]

    xtest1 = data_test[1]
    xtest = np.array(xtest1)
    ytest = data_test[0]

    xval = data_val[1]
    xval = np.array(xval)
    yval = data_val[0]
    
    X = np.concatenate((xtrain,xval), axis = 0)
    y = np.concatenate((ytrain,yval), axis = 0)
    
    return xtrain,ytrain,xtest,xtest1,ytest,xval,yval,X,y
 

#need to add F1-Score (High/Low)
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

if __name__=="__main__":
    pipeline = Pipeline([
        ('CountVec', CountVectorizer(max_features = 50000, binary=True)),
        ('clf', CLFSwitcher())
        ])
    
    parameters = [
        {
            'clf__estimator': [LinearRegression()],
            'CountVec__ngram_range': [(1,1),(1,2),(1,3)],
            'clf__estimator__normalize': (True,False)
                }]
    
    
    
    files = ['TrustPhys_','SubjectiveLit_',
            'Numeracy_','Anxiety_']

    cv = ['1','2','3','4','5']
    
    
    for filename in files:
        
        for i in cv:
            
            MeanSquaredError = []
            MeanAbsoluteError = []
            MeanAbsolutePercentageError = []
            RootMeanSquaredError = []
            PearsonR = []
            best_params = []
            
            xtrain,ytrain,xtest,xtest1,ytest,xval,yval,X,y = load_data(i, filename)
            train_indices = np.full(xtrain.shape[0],-1,dtype=int)
            val_indices = np.full(xval.shape[0],0,dtype=int)
            val_fold = np.append(train_indices,val_indices)
            pds = PredefinedSplit(test_fold = val_fold)
            
            gs = GridSearchCV(pipeline, parameters, error_score=np.nan, cv=pds, n_jobs=-1)
            clf = gs.fit(X,y)
            y_pred = clf.predict(xtest)
            MSE, MAE, MAPE, RMSE, PR = getMetrics(ytest,y_pred)
            
            best_params.append(gs.best_params_)
            MeanSquaredError.append(MSE)
            RootMeanSquaredError.append(RMSE)
            MeanAbsoluteError.append(MAE)
            MeanAbsolutePercentageError.append(MAPE)
            PearsonR.append(PR)
           
            
            
            results = {
                    'Best Parameters': best_params,
                    'Mean_Squared_Error': MeanSquaredError,
                    'Root_Mean_Squared_Error': RootMeanSquaredError,
                    'Mean_Absolute_Error': MeanAbsoluteError,
                    'Mean_Absolute_Percentage_Error': MeanAbsolutePercentageError,
                    'PearsonR': PearsonR
                    }
            
            predictions_dictionary = {
                    'sentence': xtest1,
                    'pred': y_pred
                    }
            
            #print('Done with {}'.format(filename))
            
            
#            results_df = pd.DataFrame.from_dict(results)
#            results_string = 'Results/LinRegBinary/CVCon/{}_{}resultsdf.csv'.format(i, filename)
#            results_df.to_csv(results_string, index = False)
#            
#            predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
#            predictions_df.to_csv('Results/LinRegBinary/LinRegPredictions/{}_{}_Conpredictions.csv'.format(i, filename), index=False)
#            
#            
            
