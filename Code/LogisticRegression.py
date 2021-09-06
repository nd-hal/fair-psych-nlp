

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, roc_curve, recall_score

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
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)
    
def load_data(i, filename):
    string_train = 'MedianCV/{}/{}train.txt'.format(i, filename)
    string_test = 'MedianCV/{}/{}test.txt'.format(i, filename)
    string_val = 'MedianCV/{}/{}val.txt'.format(i, filename)
    data_train = pd.read_csv(string_train, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    data_test = pd.read_csv(string_test, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    data_val = pd.read_csv(string_val, header = None, sep = '\t',encoding='ISO-8859-1').dropna()
    binary_mapper = {-1: 0}
    
    xtrain = data_train[1]
    xtrain = np.array(xtrain)
    ytrain = data_train[0]
    ytrain = ytrain.replace(binary_mapper)

    xtest1 = data_test[1]
    xtest = np.array(xtest1)
    ytest = data_test[0]
    ytest = ytest.replace(binary_mapper)

    xval = data_val[1]
    xval = np.array(xval)
    yval = data_val[0]
    yval = yval.replace(binary_mapper)
    
    X = np.concatenate((xtrain,xval), axis = 0)
    y = np.concatenate((ytrain,yval), axis = 0)
    
    return xtrain,ytrain,xtest,xtest1,ytest,xval,yval,X,y
 

#need to add F1-Score (High/Low)
def getMetrics(trueLabels, predictedLabels, predictedLabelsProbabilities):
    """Takes as input true labels, predictions, and prediction confidence scores and computes all metrics"""
    predictedLabelsProbabilities = [item[1] for item in predictedLabelsProbabilities]
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

if __name__=="__main__":
    pipeline = Pipeline([
        ('CountVec', CountVectorizer(max_features = 50000, binary=True)),
        ('clf', CLFSwitcher())
        ])
    
    parameters = [
        {
            'clf__estimator': [LogisticRegression(max_iter=4000)],
            'CountVec__ngram_range': [(1,1),(1,2),(1,3)],
            'clf__estimator__penalty': ['l1'],
            'clf__estimator__solver': ['liblinear','saga']
                },
        {
            'clf__estimator': [LogisticRegression(max_iter=4000)],
            'CountVec__ngram_range': [(1,1),(1,2),(1,3)],
            'clf__estimator__penalty': ['l2'],
            'clf__estimator__solver': ['newton-cg','lbfgs','sag']    
                }]
    

    
    files = ['TrustPhys_','SubjectiveLit_',
            'Numeracy_','Anxiety_']

    cv = ['1','2','3','4','5']
    
    
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
            best_params = []
            
            
            xtrain,ytrain,xtest,xtest1,ytest,xval,yval,X,y = load_data(i, filename)
            train_indices = np.full(xtrain.shape[0],-1,dtype=int)
            val_indices = np.full(xval.shape[0],0,dtype=int)
            val_fold = np.append(train_indices,val_indices)
            pds = PredefinedSplit(test_fold = val_fold)
            
            gs = GridSearchCV(pipeline, parameters, error_score=np.nan, cv=pds, n_jobs=-1)
            clf = gs.fit(X,y)
            y_pred_proba = clf.predict_proba(xtest)
            y_pred = clf.predict(xtest)
            accuracy, auc, precision1, precision2, recall1, recall2, fpr, tpr, f1_score1, f1_score2 = getMetrics(ytest,y_pred,y_pred_proba)
            
            
            best_params.append(gs.best_params_)
            Accuracy.append(accuracy)
            AUCScore.append(auc)
            precisionLow.append(precision1)
            precisionHigh.append(precision2)
            recallLow.append(recall1)
            recallHigh.append(recall2)
            f1_scoreLow.append(f1_score1)
            f1_scoreHigh.append(f1_score2)
            
            
            results = {
                    'Best Parameters': best_params,
                    'AUCScore': AUCScore,
                    'Accuracy': Accuracy,
                    'PrecisionHigh': precisionHigh,
                    'PrecisionLow': precisionLow,
                    'RecallHigh': recallHigh,
                    'RecallLow': recallLow,
                    'F1_ScoreLow': f1_scoreLow,
                    'F1_ScoreHigh': f1_scoreHigh
                    }
            
            predictions_dictionary = {
                    'label': y_pred,
                    'sentence': xtest1,
                    'probs': y_pred_proba[:,1]
                    }
            
            #print('Done with {}'.format(filename))
            
            
#            results_df = pd.DataFrame.from_dict(results)
#            results_string = 'Results/LogRegBinary/CVBinary/{}_{}resultsdf.csv'.format(i, filename)
#            predictions_df = pd.DataFrame.from_dict(predictions_dictionary)
#            results_df.to_csv(results_string, index = False)
#            predictions_df.to_csv('Results/LogRegBinary/LogRegPredictions/{}_{}_Binarypredictions.csv'.format(i, filename), index=False)