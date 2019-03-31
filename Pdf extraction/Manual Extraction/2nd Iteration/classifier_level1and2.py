# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:00:19 2019

@author: Lenovo
"""
import numpy as np
import nltk
import itertools
import time
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from nltk.corpus import stopwords
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer , LancasterStemmer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from prettytable import PrettyTable
from tabulate import tabulate
from sklearn import tree, svm, neighbors
import sqlite3
from sqlite3 import Error
 
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier,LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def read_data(file):
    X = []
    Y = []
    Z = []
    T = []
    R = []
    with open(file, encoding='utf-8-sig') as f:
        for line in f:
            data = line.strip().split('|')
         
            X.append(data[0])
            Y.append(data[1])
            Z.append(data[2])
            T.append(data[3])
            R.append(data[3])
            
            
    return X, Y, Z, T, R

def stem(sentence):
    
    token_words = word_tokenize(sentence)
    
    stem_sentence=[]
    
    for word in token_words:
        stem_sentence.append(PorterStemmer().stem(word))
        stem_sentence.append(" ")
   
    return "".join(stem_sentence)


def lemma(sentence):
    token_words = word_tokenize(sentence)
    
    lemma_sentence = []
    
    for word in token_words:
        lemma_sentence.append(WordNetLemmatizer().lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def is_digit(word):
    try:
        int(word)
        return True
    except ValueError:
        return False

def remove_digits(sentence):
    token_words = word_tokenize(sentence)
    
    digitless_sentence =  []
    
    for word in token_words:
        if not is_digit(word):
            digitless_sentence.append(word)
            digitless_sentence.append(" ")
    
    return "".join(digitless_sentence)

def remove_punctuations(sentence):
    sentence = sentence.translate(str.maketrans("","", string.punctuation))
    return sentence



def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None


def search_words_in_specific_categories(doc, category):
    X, Y, Z = read_data(doc)
    database = "witt.db"
    conn = create_connection(database)
    res = []
    with conn:
        for sentence in X:
            #print(sentence)
            token_words = word_tokenize(sentence.lower())
            cur = conn.cursor()
            for word in token_words: 
                cur.execute("SELECT Tag, Category from Categories WHERE Category=? and Tag=?", (category,word))
                rows = cur.fetchall()
                if(len(rows)>0):
                    for row in rows:
                        #print(row)
                        res.append(row[0])
    return set(res)


def get_tags(doc):
    results = []    
    X, Y, Z = read_data(doc)
    database = "witt.db"
    conn = create_connection(database)
    res = []
    for sentence in X:
        token_words = word_tokenize(sentence)
        cur = conn.cursor()
        for word in token_words:
            cur.execute("SELECT Category FROM Categories WHERE Tag=?", (word,))
            rows = cur.fetchall()
            word_results=[]
            if(len(rows)>0):
                for row in rows: 
                    word_results.append((word,row[0]))
                results.append(word_results)
    return results







def show_conf_matrix(Ytest, Yprediction):
    cnf_matrix = confusion_matrix(Ytest, Yprediction, labels=['UI design','data','development','document organisation','issues','requirements','source code','testing','uncertain','use case'])
    
    print(cnf_matrix)
    # the ordering of the labels is gotten by running: print('labels: ', pipeline.classes_)
    labels=['UI design','data','development','document organisation','issues','requirements','source code','testing','uncertain','use case']
    
    pyplot.figure()
    pyplot.imshow(cnf_matrix, interpolation='nearest', cmap=pyplot.cm.Blues)
    pyplot.title('Confusion Matrix')
    pyplot.colorbar()
    tick_marks = np.arange(10)
    pyplot.xticks(tick_marks, labels,rotation=90)
    pyplot.yticks(tick_marks, labels)
    
    fmt = 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        pyplot.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.tight_layout()

    pyplot.show()

def get_top_features_for_each_class(pipeline):
    labels=['UI design','data','development','document organisation','issues','requirements','source code','testing','uncertain','use case']
    classifier = pipeline.named_steps['logisticregression']
    vectorizer = pipeline.named_steps['countvectorizer']
    feature_names = vectorizer.get_feature_names()
    
    t = PrettyTable()
    
    print('')
    print('-- Top Features for each class --')
    top10s = []
    for i, label in enumerate(labels):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        top10features = []
        for j in top10:
            top10features.insert(0,feature_names[j])
        top10s.append(top10features)
        t.add_column(label,top10features)
    print(t)
    
    

##TRAINED ON ALL RELEVANT LEVEL DATA, BUT ONLY VALIDATED ON THE TEST+VALIDATION DATA FROM LEVEL 1
def second_level_classifier(X, Y, Z, Ypred, Xvalidate, Yvalidate, Zvalidate, level): ## REMOVE Y FROM INPUT LIST - it's not used
    ##Inputs:
    # X: all sentences to train on
    # Z: the true labels of the sentences used for training
    # Ypred: 1st level label prediction of validation data
    # Yvalidate: 1st level true label of validation data
    # Xvalidate: sentences to predict
    # Zvalidate: labels to predict
    # level: 

    sentences = []
    Xval = []
    Yval = []
    Zval = []
    
    
    
    for i, label in enumerate(Ypred):
        if(Ypred[i]==Yvalidate[i]):           #Exclude sentences from validation data that have been misclassified
            if(label == level):
                sentences.append(i)    
    
    #Create the relevant data using the pos numbers from sentences.
    for sent in sentences:
        Xval.append(Xvalidate[sent])
        Yval.append(Yvalidate[sent])
        Zval.append(Zvalidate[sent])

        
    
    Xtrain, Xtest, Ztrain, Ztest = train_test_split(X, Z, test_size=0.25, random_state=42)
    
    
    print('')
    print('')
    print('----------------------------------------------------')
    print('')
    print("---------- %s classifier ----------" %level)
    print('Number of training instances: ', len(Xtrain))
    print('Number of testing instances: ', len(Xtest))
    print('Number of validation instances: ', len(Xval))
        
    pipeline = make_pipeline(
            TfidfVectorizer(max_features=500, analyzer='word', lowercase=True, ngram_range=(1,2),
                        norm='l2', sublinear_tf=True),
            ##REQUIREMENTS##
            #BernoulliNB(alpha=2, binarize=0, fit_prior=True, class_prior=None)                
            #DecisionTreeClassifier(max_features=50, class_weight='balanced',min_impurity_split=0.5, random_state=42)
            #ExtraTreeClassifier(max_features=100, class_weight='balanced',min_impurity_split=0.6, random_state=42)
            #ExtraTreesClassifier(max_features=100, class_weight='balanced',min_impurity_split=0.7, random_state=42)
            #KNeighborsClassifier(n_neighbors=25)
            #LinearSVC(C=0.1, class_weight='balanced', dual=True, fit_intercept=True,intercept_scaling=1, loss='hinge', multi_class='ovr', penalty='l2', tol=1e-05)
            #LogisticRegression(n_jobs=1, multi_class='multinomial', class_weight='balanced', solver='lbfgs', C=0.01, penalty='l2')
            #LogisticRegressionCV(n_jobs=-1, multi_class='multinomial', class_weight='balanced', cv=5,Cs=1, random_state=42)
            #MLPClassifier(early_stopping=True, random_state=42)
            NearestCentroid()
            #RadiusNeighborsClassifier(radius=100)
            #RandomForestClassifier(class_weight='balanced',min_impurity_split=0, max_depth=7, random_state=42 )
            #RidgeClassifier(alpha=6, random_state=42)
            #RidgeClassifierCV(alphas=[10], class_weight='balanced')
            #SGDClassifier(alpha=0.5, loss="log", penalty="l2", class_weight='balanced', n_iter_no_change=3,early_stopping=True, random_state=42)
            
            ##SYSTEM##
            #BernoulliNB(alpha=2, binarize=0, fit_prior=True, class_prior=None)                
            #DecisionTreeClassifier(max_features=50, class_weight='balanced',min_impurity_split=0.5, random_state=42)
            #ExtraTreeClassifier(max_features=50, class_weight='balanced',min_impurity_split=0.5, random_state=42)
            #ExtraTreesClassifier(max_features=50, class_weight='balanced',min_impurity_split=0.5, random_state=42)
            #KNeighborsClassifier(n_neighbors=25)
            #LinearSVC(C=0.05, class_weight='balanced', dual=True, fit_intercept=True,intercept_scaling=1, loss='hinge', multi_class='ovr', penalty='l2', tol=1e-05)
            #LogisticRegression(n_jobs=1, multi_class='multinomial', class_weight='balanced', solver='lbfgs', C=0.01, penalty='l2')
            #LogisticRegressionCV(n_jobs=-1, multi_class='multinomial', class_weight='balanced', cv=5,Cs=1, random_state=42)
            #MLPClassifier(early_stopping=True, random_state=42)
            ##NearestCentroid()
            #RadiusNeighborsClassifier(radius=100)
            #RandomForestClassifier(class_weight='balanced',min_impurity_split=0, max_depth=4, random_state=42 )
            #RidgeClassifier(alpha=10, random_state=42)
            #RidgeClassifierCV(alphas=[10], class_weight='balanced')
            #SGDClassifier(alpha=0.5, loss="log", penalty="l2", class_weight='balanced', n_iter_no_change=3,early_stopping=True, random_state=42)
    )
    pipeline.fit(Xtrain, Ztrain)
    Zpred = pipeline.predict(Xtest)    
    print('')
    print('%s train Accuracy: ' %level, accuracy_score(Ztrain, pipeline.predict(Xtrain)))    
    print('%s Overall test Accuracy: ' %level, accuracy_score(Ztest, Zpred))    
    print('')
    print('----- VALIDATION SET ACCURACY---------')
    valPred = pipeline.predict(Xval)
    print('')
    print('%s Accuracy: ' %level, accuracy_score(Zval, valPred))    
    print('')

    
    if level == 'requirement': 
        generalTrue, generalPred, functionalTrue, functionalPred, nonFunctionalTrue, nonFunctionalPred, featureTrue, featurePred, useCaseTrue, useCasePred = categorySplitReq(Zval, valPred)
        print('')
        print('samples:', len(generalTrue))
        print('Req-General Accuracy:  ', accuracy_score(generalTrue, generalPred))    
        print('')
        print('samples:', len(functionalTrue))
        print('Functionality&behaviour Accuracy:  ', accuracy_score(functionalTrue, functionalPred))    
        print('')
        print('samples:', len(nonFunctionalTrue))
        print('non-functional Accuracy:  ', accuracy_score(nonFunctionalTrue, nonFunctionalPred))    
        print('')
        print('samples:', len(featureTrue))
        print('Feature Accuracy:  ', accuracy_score(featureTrue, featurePred))    
        print('')
        print('samples:', len(useCaseTrue))
        print('Use case Accuracy:  ', accuracy_score(useCaseTrue, useCasePred))    
        print('')
        
    if level == 'system':
        structureTrue, structurePred, behaviourTrue, behaviourPred, dataTrue, dataPred, uiDesignTrue, uiDesignPred = categorySplitSystem(Zval, valPred)
        print('')
        print('samples:', len(structureTrue))
        print('Structure Accuracy:  ', accuracy_score(structureTrue, structurePred))    
        print('')
        print('samples:', len(behaviourTrue))
        print('Behaviour Accuracy:  ', accuracy_score(behaviourTrue, behaviourPred))    
        print('')
        print('samples:', len(dataTrue))
        print('data Accuracy:  ', accuracy_score(dataTrue, dataPred))    
        print('')
        print('samples:', len(uiDesignTrue))
        print('UI design Accuracy:  ', accuracy_score(uiDesignTrue, uiDesignPred))    
        print('')
        
    

def categorySplitSystem(test, pred):
    structureTrue = []
    structurePred = []
    
    behaviourTrue = []
    behaviourPred = []
    
    dataTrue = []
    dataPred = []
    
    uiDesignTrue = []
    uiDesignPred = []
    
    for i, cat in enumerate(test):
        if cat == 'structure':
            structureTrue.append(test[i])
            structurePred.append(pred[i])
        elif cat == 'behaviour':
            behaviourTrue.append(test[i])
            behaviourPred.append(pred[i])
        elif cat == 'data':
            dataTrue.append(test[i])
            dataPred.append(pred[i])
        elif cat == 'ui design':
            uiDesignTrue.append(test[i])
            uiDesignPred.append(pred[i])
    
    return structureTrue, structurePred, behaviourTrue, behaviourPred, dataTrue, dataPred, uiDesignTrue, uiDesignPred
    
def categorySplitReq(test, pred):
    generalTrue = []
    generalPred = []
    
    functionalTrue = []
    functionalPred = []
    
    nonFunctionalTrue = []
    nonFunctionalPred = []
    
    featureTrue = []
    featurePred = []
    
    useCaseTrue = []
    useCasePred = []
    
    for i, cat in enumerate(test):
        if cat == 'r-general':
            generalTrue.append(test[i])
            generalPred.append(pred[i])
        elif cat == 'functionality & behaviour':
            functionalTrue.append(test[i])
            functionalPred.append(pred[i])
        elif cat == 'non-functional':
            nonFunctionalTrue.append(test[i])
            nonFunctionalPred.append(pred[i])
        elif cat == 'feature':
            featureTrue.append(test[i])
            featurePred.append(pred[i])
        elif cat == 'use case':
            useCaseTrue.append(test[i])
            useCasePred.append(pred[i])
    
    return generalTrue, generalPred, functionalTrue, functionalPred, nonFunctionalTrue, nonFunctionalPred, featureTrue, featurePred, useCaseTrue, useCasePred
    
def categorySplitLevel1(test, pred):
    requirementTrue = []
    requirementPred = []
    
    systemTrue = []
    systemPred = []
    
    domainTrue = []
    domainPred = []
    
    devProcessTrue = []
    devProcessPred = []
    
    docOrgTrue = []
    docOrgPred = []
    
    uncertainTrue = []
    uncertainPred = []
    
    nonInformationTrue = []
    nonInformationPred = []
    
    for i, cat in enumerate(test):
        if cat == 'requirement': 
            requirementTrue.append(test[i])
            requirementPred.append(pred[i])
        elif cat == 'system':
            systemTrue.append(test[i])
            systemPred.append(pred[i])
        elif cat == 'domain':
            domainTrue.append(test[i])
            domainPred.append(pred[i])
        elif cat == 'development process':
            devProcessTrue.append(test[i])
            devProcessPred.append(pred[i])
        elif cat == 'document organisation':
            docOrgTrue.append(test[i])
            docOrgPred.append(pred[i])
        elif cat == 'system':
            systemTrue.append(test[i])
            systemPred.append(pred[i])
        elif cat == 'uncertain':
            uncertainTrue.append(test[i])
            uncertainPred.append(pred[i])
        elif cat == 'non-information':
            nonInformationTrue.append(test[i])
            nonInformationPred.append(pred[i])
            
    return requirementTrue, requirementPred, systemTrue, systemPred, domainTrue, domainPred, devProcessTrue, devProcessPred, docOrgTrue, docOrgPred, uncertainTrue, uncertainPred, nonInformationTrue, nonInformationPred

def get_relevant_section(labels, section):
    
    locations=[]
    for i, label in enumerate(labels):
      if label == section:
          locations.append(i)
          
    return locations

    
    
    
        
def train_classifier():
    X, Y, Z, T, R = read_data('../CSV Data/FinalData.csv')   

    for pos, sentence in enumerate(X):
        sentence = lemma(sentence)
        sentence = stem(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        X[pos] = sentence
        
    Xtrain, Xtest, Ytrain, Ytest, Ztrain, Ztest, Ttrain, Ttest, Rtrain, Rtest = train_test_split(X, Y, Z, T, R, test_size=0.25, random_state=42)
    Xtrain, Xval, Ytrain, Yval, Ztrain, Zval, Ttrain, Tval, Rtrain, Rval = train_test_split(Xtrain, Ytrain, Ztrain, Ttrain, Rtrain, test_size=0.1, random_state=42)
    stopWords = set(stopwords.words('english'))

    
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words=stopWords, max_features=500, analyzer='word', lowercase=True, ngram_range=(1,2),
                        norm='l2', sublinear_tf=True),
        #BernoulliNB(alpha=1, binarize=0, fit_prior=True, class_prior=None)                
        #DecisionTreeClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.8)
        #ExtraTreeClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.8)
        #ExtraTreesClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.8)
        #KNeighborsClassifier(n_neighbors=100)
        LinearSVC(C=0.01, class_weight='balanced', dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=500, multi_class='ovr', penalty='l2', tol=1e-05)
        #LogisticRegression(n_jobs=1, multi_class='multinomial', class_weight='balanced', solver='lbfgs', C=0.1, penalty='l2')
        #LogisticRegressionCV(n_jobs=-1, multi_class='multinomial', class_weight='balanced', cv=5,Cs=1)
        #MLPClassifier(early_stopping=True, random_state=42)
        #NearestCentroid()
        #RadiusNeighborsClassifier(radius=15)
        #RandomForestClassifier(class_weight='balanced',min_impurity_split=0, max_depth=4, random_state=0 )
        #RidgeClassifier(alpha=10, random_state=0, normalize=True)
        #RidgeClassifierCV(alphas=[5,6,7], normalize=True, class_weight='balanced')
        #SGDClassifier(alpha=0.01, loss="log", penalty="l2", class_weight='balanced', n_iter_no_change=3,early_stopping=True) #63,5 - 63
        
        
    )
    print('-------- CROSS VALIDATION -----------')
    
    scores = cross_val_score(pipeline, Xtrain+Xtest+Xval, Ytrain+Ytest+Yval, cv=5)
    print(scores)
    print(np.average(scores))

    pipeline.fit(Xtrain, Ytrain)
    Yprediction = pipeline.predict(Xtest)

    print('----- VALIDATION SET ACCURACY---------')
    valPred = pipeline.predict(Xval)
    print('')
    print('Accuracy: ', accuracy_score(Yval, valPred))    
    print('')    
    
    requirementTrue, requirementPred, systemTrue, systemPred, domainTrue, domainPred, devProcessTrue, devProcessPred, docOrgTrue, docOrgPred, uncertainTrue, uncertainPred, nonInformationTrue, nonInformationPred = categorySplitLevel1(Yval, valPred)
    print('')
    print('samples:', len(requirementTrue))
    print('Requirement Accuracy:  ', accuracy_score(requirementTrue, requirementPred))    
    print('')
    print('samples:', len(systemTrue))
    print('System Accuracy:  ', accuracy_score(systemTrue, systemPred))    
    print('')
    print('samples:', len(domainTrue))
    print('Domain Accuracy:  ', accuracy_score(domainTrue, domainPred))    
    print('')
    print('samples:', len(devProcessTrue))
    print('Dev Process Accuracy:  ', accuracy_score(devProcessTrue, devProcessPred))    
    print('')
    print('samples:', len(docOrgTrue))
    print('Doc Organisation Accuracy:  ', accuracy_score(docOrgTrue, docOrgPred))    
    print('')
    print('samples:', len(uncertainTrue))
    print('Uncertain Accuracy:  ', accuracy_score(uncertainTrue, uncertainPred))    
    print('')
    print('samples:', len(nonInformationTrue))
    print('Non-info Accuracy:  ', accuracy_score(nonInformationTrue, nonInformationPred))    
    print('')
    print('---------------')
    print('')
    print('Overall train Accuracy:  ', accuracy_score(Ytrain, pipeline.predict(Xtrain) ))    
    print('')
    print('')
    print('Overall test Accuracy:  ', accuracy_score(Ytest, Yprediction))    
    print('')
    
    
    
    
    
    trainLocations = get_relevant_section(Ytrain, 'requirement')
    testLocations = get_relevant_section(Ytest, 'requirement')
    valLocations = get_relevant_section(Yval,'requirement')
    
    XtrainSection = []; YtrainSection = []; ZtrainSection = []
    XtestSection = []; YtestSection = []; ZtestSection = []; YpredictionSection = []
    XvalSection = []; YvalSection = []; ZvalSection = []; valPredSection = []
    for num in trainLocations:
        XtrainSection.append(Xtrain[num])
        YtrainSection.append(Ytrain[num])
        ZtrainSection.append(Ztrain[num])
    for num in testLocations:
        XtestSection.append(Xtest[num])
        YtestSection.append(Ytest[num])
        ZtestSection.append(Ztest[num])
        YpredictionSection.append(Yprediction[num])
        
    for num in valLocations:
        XvalSection.append(Xval[num])
        YvalSection.append(Yval[num])
        ZvalSection.append(Zval[num])
        valPredSection.append(valPred[num])


    second_level_classifier(XtrainSection, YtrainSection, ZtrainSection, YpredictionSection+valPredSection, XtestSection+XvalSection, YtestSection+YvalSection, ZtestSection+ZvalSection, 'requirement')
    
    trainLocations = get_relevant_section(Ytrain, 'system')
    testLocations = get_relevant_section(Ytest, 'system')
    valLocations = get_relevant_section(Yval,'system')
    
    

    
    
    
    XtrainSection = []; YtrainSection = []; ZtrainSection = []
    XtestSection = []; YtestSection = []; ZtestSection = []; YpredictionSection = []
    XvalSection = []; YvalSection = []; ZvalSection = []; valPredSection = []
    for num in trainLocations:
        XtrainSection.append(Xtrain[num])
        YtrainSection.append(Ytrain[num])
        ZtrainSection.append(Ztrain[num])
    for num in testLocations:
        XtestSection.append(Xtest[num])
        YtestSection.append(Ytest[num])
        ZtestSection.append(Ztest[num])
        YpredictionSection.append(Yprediction[num])
        
    for num in valLocations:
        XvalSection.append(Xval[num])
        YvalSection.append(Yval[num])
        ZvalSection.append(Zval[num])
        valPredSection.append(valPred[num])
        
    second_level_classifier(XtrainSection, YtrainSection, ZtrainSection, YpredictionSection+valPredSection, XtestSection+XvalSection, YtestSection+YvalSection, ZtestSection+ZvalSection, 'system')
    
    
    #second_level_classifier(Xtest, Ytest, Yprediction, Ztest, 'development process')
    
    
    
train_classifier()



    
    
    
    
    
    
    
    
    
    

