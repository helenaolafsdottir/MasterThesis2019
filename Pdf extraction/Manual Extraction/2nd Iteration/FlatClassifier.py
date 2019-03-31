# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
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
            R.append(data[4])
            
            
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
    
    
    
def second_level_classifier(Xtest, Ytest, Yprediction, Ztest, level):
    
    reqs = []
    X = []
    Y = []
    Z = []
    for i, label in enumerate(Yprediction):
        if(Yprediction[i]==Ytest[i]):           #Exclude sentences that have been misclassified
            if(label == level):
                reqs.append(i)    
    
    #Create the relevant data using the pos numbers from reqs.
    for req in reqs:
        X.append(Xtest[req])
        Y.append(Yprediction[req])
        Z.append(Ztest[req])
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Z, test_size=0.2, random_state=42)
    
    print('')
    print('')
    print("---------- %s classifier ----------" %level)
    print('Number of training instances: ', len(Xtrain))
    print('Number of testing instances: ', len(Xtest))
    
    pipeline = make_pipeline(
            CountVectorizer(max_features=5500, analyzer='word', ngram_range=(1,2)),
            TfidfTransformer(norm='l2', sublinear_tf=True),
            LogisticRegression(n_jobs=1, C=1e3)
            
    )
    pipeline.fit(Xtrain, Ytrain)
    Ypred = pipeline.predict(Xtest)    
    print('')
    print('%s accuracy: ' %level, accuracy_score(Ytest, Ypred))    
    print('')
    

def identify_user_stories(Xtest, Ytest, Ztest):
    
    X = []
    Y = []
    Z = []
    for i, label in enumerate(Ytest):
        if(label == 'requirements'): # We are only interested in requirements
            X.append(Xtest[i])
            Y.append(Ytest[i])
            Z.append(Ztest[i])

    print('--- User Stories ---')
    
    for i, sentence in enumerate(X):
        if("As a" in sentence): 
            print('Label: ', Z[i])
            print('User story: ',sentence, '\n')
            
        

    
def get_data_statistics():
    X, Y, Z = read_data('../CSV Data/FinalData.csv')
 
    # the ordering of the labels is gotten by running: print('labels: ', pipeline.classes_)
    labels=['UI design','data','development','document organisation','issues','requirements','source code','testing','uncertain','use case']
    
    data = (Y.count('requirements'), Y.count('development'), Y.count('use case'), Y.count('UI design'), Y.count('data'), Y.count('testing'), Y.count('document organisation'),Y.count('source code'),Y.count('issues'),Y.count('uncertain'))
    
    #Crate barplot
    pyplot.bar(labels, data)
    pyplot.xticks(rotation='vertical')
    pyplot.title('AK categories')
    pyplot.xlabel('Category')
    pyplot.ylabel('Number of sentences')
    for a,b in zip(labels, data):
        pyplot.text(a, b, str(b))
    pyplot.show()

def categorySplit(chunk1, chunk2):
    
    stakeholderTrue = []
    stakeholderPred = []
    
    rGeneralTrue = []
    rGeneralPred = []
    functionalityBehaviourTrue = []
    functionalityBehaviourPred = []
    nonFunctionalBehaviourTrue = []
    nonFunctionalBehaviourPred = []
    featureTrue = []
    featurePred = []
    useCaseTrue = []
    useCasePred = []
       
    sArchitectureTrue = []
    sArchitecturePred = []
    sTechSolTrue = []
    sTechSolPred = []
    sSourceCodeTrue = []
    sSourceCodePred = []
    bTechSolTrue = []
    bTechSolPred = []
    bGeneralTrue = []
    bGeneralPred = []
    dArchitectureTrue = []
    dArchitecturePred = []
    uArchitectureTrue = []
    uArchitecturePred = []
    uImplementationTrue = []
    uImplementationPred = []
    
    
    
    devPracticeTrue = []
    devPracticePred = []
    issueTrue = []
    issuePred = []
    riskTrue = []
    riskPred = []
    testingTrue = []
    testingPred = []
    
    
    docOrgTrue = []
    docOrgPred = []
    
    uncertainTrue = []
    uncertainPred = []
    
    nonInfoTrue = []
    nonInfoPred = []
    
    for i, cat in enumerate(chunk1):
        if cat == 'stakeholder':
            stakeholderTrue.append(chunk1[i])
            stakeholderPred.append(chunk2[i])
            
        elif cat == 'r-general': 
            rGeneralTrue.append(chunk1[i])
            rGeneralPred.append(chunk2[i])
        elif cat == 'functionality & behaviour': 
            functionalityBehaviourTrue.append(chunk1[i])
            functionalityBehaviourPred.append(chunk2[i])
        elif cat == 'non-functional': 
            nonFunctionalBehaviourTrue.append(chunk1[i])
            nonFunctionalBehaviourPred.append(chunk2[i])
        elif cat == 'feature': 
            featureTrue.append(chunk1[i])
            featurePred.append(chunk2[i])
        elif cat == 'use case': 
            useCaseTrue.append(chunk1[i])
            useCasePred.append(chunk2[i])
        
        elif cat == 's-architecture':
            sArchitectureTrue.append(chunk1[i])
            sArchitecturePred.append(chunk2[i])
        elif cat == 's-techsolution':
            sTechSolTrue.append(chunk1[i])
            sTechSolPred.append(chunk2[i])
        elif cat == 's-sourcecode':
            sSourceCodeTrue.append(chunk1[i])
            sSourceCodePred.append(chunk2[i])
        elif cat == 'b-techsolution':
            bTechSolTrue.append(chunk1[i])
            bTechSolPred.append(chunk2[i])
        elif cat == 'b-general':
            bGeneralTrue.append(chunk1[i])
            bGeneralPred.append(chunk2[i])
        elif cat == 'd-architecture':
            dArchitectureTrue.append(chunk1[i])
            dArchitecturePred.append(chunk2[i])
        elif cat == 'u-architecture':
            uArchitectureTrue.append(chunk1[i])
            uArchitecturePred.append(chunk2[i])
        elif cat == 'u-implementation':
            uImplementationTrue.append(chunk1[i])
            uImplementationPred.append(chunk2[i])
        
        elif cat == 'development practice':
            devPracticeTrue.append(chunk1[i])
            devPracticePred.append(chunk2[i])
        elif cat == 'issue':
            issueTrue.append(chunk1[i])
            issuePred.append(chunk2[i])
        elif cat == 'issue':
            riskTrue.append(chunk1[i])
            riskPred.append(chunk2[i])
        elif cat == 'testing':
            testingTrue.append(chunk1[i])
            testingPred.append(chunk2[i])
            
            
        elif cat == 'document organisation':
            docOrgTrue.append(chunk1[i])
            docOrgPred.append(chunk2[i])
        elif cat == 'uncertain':
            uncertainTrue.append(chunk1[i])
            uncertainPred.append(chunk2[i])
        elif cat == 'non-information':
            nonInfoTrue.append(chunk1[i])
            nonInfoPred.append(chunk2[i])
            
    return stakeholderTrue, stakeholderPred, rGeneralTrue, rGeneralPred, functionalityBehaviourTrue, functionalityBehaviourPred, nonFunctionalBehaviourTrue, nonFunctionalBehaviourPred, featureTrue, featurePred, useCaseTrue, useCasePred, sArchitectureTrue, sArchitecturePred, sTechSolTrue, sTechSolPred, sSourceCodeTrue, sSourceCodePred, bTechSolTrue, bTechSolPred, bGeneralTrue, bGeneralPred, dArchitectureTrue, dArchitecturePred, uArchitectureTrue, uArchitecturePred, uImplementationTrue, uImplementationPred, devPracticeTrue, devPracticePred, issueTrue, issuePred, riskTrue, riskPred, testingTrue, testingPred, docOrgTrue, docOrgPred, uncertainTrue, uncertainPred, nonInfoTrue, nonInfoPred
    
def train_classifier():
    X, Y, Z, T, R = read_data('../CSV Data/FinalData.csv')   

    for pos, sentence in enumerate(X):
        sentence = lemma(sentence)
        sentence = stem(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        X[pos] = sentence
        
    Xtrain, Xtest, Ytrain, Ytest, Ztrain, Ztest, Ttrain, Ttest, Rtrain, Rtest = train_test_split(X, Y, Z, T, R, test_size=0.25, random_state=42)
    Xtrain, Xval, Ytrain, Yval, Ztrain, Zval, Ttrain, Tval, Rtrain, Rval = train_test_split(Xtrain, Ytrain, Ztrain, Ttrain, Rtrain, test_size=0.15, random_state=42)
    stopWords = set(stopwords.words('english'))


    pipeline = make_pipeline(
        #CountVectorizer(stop_words=stopWords, max_features=5000, analyzer='word', lowercase=True, ngram_range=(1,3)),
        #TfidfTransformer(norm='l2', sublinear_tf=True),
        #Normalizer(),
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        #OneVsOneClassifier(LinearSVC(random_state=0))     #57,8 - 58,3
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        #OneVsRestClassifier(LinearSVC(random_state=42, penalty='l2', loss='hinge'))     #59,5 - 59,5
        
        TfidfVectorizer(stop_words=stopWords, max_features=500, analyzer='word', lowercase=True, ngram_range=(1,2),
                        norm='l2', sublinear_tf=True),
        BernoulliNB(alpha=2)
        #DecisionTreeClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.9)
        #ExtraTreeClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.9)
        #ExtraTreesClassifier(max_features=500, class_weight='balanced',min_impurity_split=0.9)
        #KNeighborsClassifier(n_neighbors=100)
        #LinearSVC(C=0.01, class_weight='balanced', dual=True, fit_intercept=True, loss='squared_hinge', multi_class='ovr', penalty='l2')
        #LogisticRegression(n_jobs=1, multi_class='multinomial', class_weight='balanced', solver='lbfgs', C=0.03, penalty='l2')
        #LogisticRegressionCV(n_jobs=-1, multi_class='multinomial', class_weight='balanced', cv=2,Cs=1)
        #MLPClassifier(early_stopping=True, random_state=42, solver = 'lbfgs', alpha = 8)
        #NearestCentroid()
        #RadiusNeighborsClassifier(radius=10)
        #RandomForestClassifier(class_weight='balanced',min_impurity_split=0, max_depth=4, random_state=0 )
        #RidgeClassifier(alpha=25, random_state=0, normalize=True)
        #RidgeClassifierCV(alphas=[30], class_weight='balanced')
        #SGDClassifier(alpha=0.5, loss="log", penalty="l2", class_weight='balanced', random_state=42) #63,5 - 63
        
        
    )
#    print('-------- CROSS VALIDATION -----------')
#    
#    scores = cross_val_score(pipeline, Xtrain+Xtest+Xval, Rtrain+Rtest+Rval, cv=5)
#    print(scores)
#    print(np.average(scores))

    pipeline.fit(Xtrain, Rtrain)
    Rprediction = pipeline.predict(Xtest)    
    
    print('---------------')
    print('')
    print('Overall train Accuracy:  ', accuracy_score(Rtrain, pipeline.predict(Xtrain) ))    
    print('')
    print('')
    print('Overall test Accuracy:  ', accuracy_score(Rtest, Rprediction))    
    print('')
    

    print('----- VALIDATION SET ACCURACY---------')
    valPred = pipeline.predict(Xval)
    print('')
    print('Accuracy: ', accuracy_score(Rval, valPred))    
    print('')
    
    stakeholderTrue, stakeholderPred, rGeneralTrue, rGeneralPred, functionalityBehaviourTrue, functionalityBehaviourPred, nonFunctionalBehaviourTrue, nonFunctionalBehaviourPred, featureTrue, featurePred, useCaseTrue, useCasePred, sArchitectureTrue, sArchitecturePred, sTechSolTrue, sTechSolPred, sSourceCodeTrue, sSourceCodePred, bTechSolTrue, bTechSolPred, bGeneralTrue, bGeneralPred, dArchitectureTrue, dArchitecturePred, uArchitectureTrue, uArchitecturePred, uImplementationTrue, uImplementationPred, devPracticeTrue, devPracticePred, issueTrue, issuePred, riskTrue, riskPred, testingTrue, testingPred, docOrgTrue, docOrgPred, uncertainTrue, uncertainPred, nonInfoTrue, nonInfoPred = categorySplit(Rval, valPred)
    

    print('samples:', len(stakeholderTrue))
    print('Stakeholder Accuracy:  ', accuracy_score(stakeholderTrue, stakeholderPred))    
    print('')
    print('samples:', len(rGeneralTrue))
    print('r-General Accuracy:  ', accuracy_score(rGeneralTrue, rGeneralPred))    
    print('')
    print('samples:', len(functionalityBehaviourTrue))
    print('functionality & behaviour Accuracy:  ', accuracy_score(functionalityBehaviourTrue, functionalityBehaviourPred))    
    print('')
    print('samples:', len(nonFunctionalBehaviourTrue))
    print('non-functional Accuracy:  ', accuracy_score(nonFunctionalBehaviourTrue,  nonFunctionalBehaviourPred))    
    print('')
    print('samples:', len(featureTrue))
    print('feature Accuracy:  ', accuracy_score(featureTrue, featurePred))    
    print('')
    print('samples:', len(useCaseTrue))
    print('use case Accuracy:  ', accuracy_score(useCaseTrue, useCasePred))    
    print('')
    print('samples:', len(sArchitectureTrue))
    print('s-Architecture Accuracy:  ', accuracy_score(sArchitectureTrue, sArchitecturePred))    
    print('')
    print('samples:', len(sTechSolTrue))
    print('s - tech solution Accuracy:  ', accuracy_score(sTechSolTrue, sTechSolPred))    
    print('')
    print('samples:', len(sSourceCodeTrue))
    print('s - source code Accuracy:  ', accuracy_score(sSourceCodeTrue, sSourceCodePred))    
    print('')
    print('samples:', len(bTechSolTrue))
    print('b-tech sol Accuracy:  ', accuracy_score(bTechSolTrue, bTechSolPred))    
    print('')
    print('samples:', len(bGeneralTrue))
    print('b - general Accuracy:  ', accuracy_score(bGeneralTrue, bGeneralPred))    
    print('')
    print('samples:', len(dArchitectureTrue))
    print('d-Architeture Accuracy:  ', accuracy_score(dArchitectureTrue, dArchitecturePred))    
    print('')
    print('samples:', len(uArchitectureTrue))
    print('u-Architecture Accuracy:  ', accuracy_score(uArchitectureTrue, uArchitecturePred))    
    print('')
    print('samples:', len(uImplementationTrue))
    print('u-Implementation Accuracy:  ', accuracy_score(uImplementationTrue, uImplementationPred))    
    print('')
    print('samples:', len(devPracticeTrue))
    print('dev Practice Accuracy:  ', accuracy_score(devPracticeTrue, devPracticePred))    
    print('')
    print('samples:', len(issueTrue))
    print('issue Accuracy:  ', accuracy_score(issueTrue, issuePred))    
    print('')
    print('samples:', len(riskTrue))
    print('risk Accuracy:  ', accuracy_score(riskTrue, riskPred))    
    print('')
    print('samples:', len(testingTrue))
    print('testing Accuracy:  ', accuracy_score(testingTrue, testingPred))    
    print('')
    print('samples:', len(docOrgTrue))
    print('document Organisation Accuracy:  ', accuracy_score(docOrgTrue, docOrgPred))    
    print('')
    print('samples:', len(uncertainTrue))
    print('uncertain Accuracy:  ', accuracy_score(uncertainTrue, uncertainPred))    
    print('')
    print('samples:', len(nonInfoTrue))
    print('non-info Accuracy:  ', accuracy_score(nonInfoTrue, nonInfoPred))    
    print('')
    
    

#    pipeline.fit(Xtrain, Ztrain)
#    Zprediction = pipeline.predict(Xtest)    
#    print('')
#    print(' Accuracy 2 levels: ', accuracy_score(Ztest, Zprediction))    
#    print('')
#    
#    pipeline.fit(Xtrain, Ttrain)
#    Tprediction = pipeline.predict(Xtest)    
#    print('')
#    print(' Accuracy 3 levels: ', accuracy_score(Ttest, Tprediction))    
#    print('')
#    pipeline.fit(Xtrain, Rtrain)
#    Rprediction = pipeline.predict(Xtest)  
#    print('')
#    print(' Accuracy 4 levels: ', accuracy_score(Rtest, Rprediction))    
#    print('')
#    
  
#    pipeline.fit(Xtrain, Rtrain)
#    Rprediction = pipeline.predict(Xtest)    
#    print('Overall train Accuracy:  ', accuracy_score(Rtrain, pipeline.predict(Xtrain) ))    
#    print('')
#    print('')
#    print('Overall test Accuracy:  ', accuracy_score(Rtest, Rprediction))    
#    print('')
    


    #show_conf_matrix(Ytest, YpredictionSGD)
  
    #labels_report=['UI design','development','document organisation','issues','requirements','source code','testing','uncertain','use case']
    #print(classification_report(Ytest, YpredictionSGD, target_names=labels_report))
    
    
    #get_top_features_for_each_class(pipeline)
  
    #second_level_classifier(Xtest, Ytest, YpredictionSGD, Ztest, 'requirements')
    #second_level_classifier(Xtest, Ytest, YpredictionSGD, Ztest, 'development')
    
    #identify_user_stories(Xtest, Ytest, Ztest)

#results = get_tags('../CSV Data/FinalData.csv')
#get_data_statistics()    
train_classifier()
#print(results)
#prog_languages = search_words_in_specific_categories('../CSV Data/FinalData.csv', 'programming-language')
#print('Programming Languages:')
#print(prog_languages)
    
    


"""
Created on Sat Mar 23 14:00:03 2019

@author: Lenovo
"""

