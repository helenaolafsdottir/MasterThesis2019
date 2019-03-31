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
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier,LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer , LancasterStemmer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from prettytable import PrettyTable
from tabulate import tabulate
from sklearn import tree, svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sqlite3
from sqlite3 import Error
 


def read_data(file):
    X = []
    Y = []
    Z = []
    with open(file, encoding='utf-8-sig') as f:
        for line in f:
            data = line.strip().split('|')
         
            X.append(data[0])
            Y.append(data[1])
            Z.append(data[2])
            
    return X, Y, Z

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
    #print(doc)
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
            
        
    
def train_classifier():
    X, Y, Z = read_data('CSV Data/snowflake_annotated_data.csv')
    
    identify_user_stories(X,Y,Z)
    
    Xtrain, Xtest, Ytrain, Ytest, Ztrain, Ztest = train_test_split(X, Y, Z, test_size=0.2, random_state=42)
    
    stopWords = set(stopwords.words('english'))
    
    
    # --------- NLP -------------     
    for pos, sentence in enumerate(Xtrain):
        #sentence = lemma(sentence)
        sentence = stem(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        Xtrain[pos] = sentence
        

    pipeline = make_pipeline(
        CountVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        TfidfTransformer(norm='l2', sublinear_tf=True),
        #Normalizer(),
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        #OneVsOneClassifier(LinearSVC(random_state=0))     #62.43
        
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        #OneVsRestClassifier(LinearSVC(random_state=42, penalty='l2', loss='hinge'))     #67.05
        
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        LogisticRegression(n_jobs=1, C=1e3)  #69.36 
        
        #SGDClassifier(alpha=0.0001, loss="hinge", penalty="elasticnet", max_iter=150) #68.80
        #OneVsRestClassifier(MultinomialNB())                #53
        #OneVsOneClassifier(MultinomialNB())                #52
    )

    
    '''
    pipeline = Pipeline([#('vect', CountVectorizer(stop_words=stopWords, analyzer='word', lowercase=True, ngram_range=(1,2))),
                         ('vect', CountVectorizer(stop_words=stopWords, analyzer='word', lowercase=True)),
                         #('vect', CountVectorizer(stop_words=stopWords, max_features=5000, analyzer='word', lowercase=True, ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer()),
                         #('feature_select', SelectKBest(k=5500)),
                         #('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                         #('clf', SGDClassifier(random_state=42)),
                         #('clf', LogisticRegression(n_jobs=1, C=1e5)),
                         #('clf', LogisticRegression(n_jobs=1, C=1e5)),
                         #('clf', MultinomialNB()),
                         ('clf', LinearSVC())
                         #('clf', OneVsRestClassifier(LinearSVC())),
                        ])
    
    parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__max_features': (None, 5000, 5500, 6000, 6500, 7500, 8000),
            'vect__max_features': (None, 5500, 6500, 7500),
            'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
            'tfidf__use_idf': (True, False),
            'clf__max_iter': (5,),
            'clf__loss':('hinge'),
            #'clf__penalty': ('l2', 'elasticnet','l1','none'),
            'clf__penalty': ('l2',),
            #'clf__alpha': (0.00001, 0.000001),
            }
    
    #CV = GridSearchCV(pipeline, parameters, scoring = 'mean_absolute_error', cv=5,n_jobs= 1, verbose=1)
    #CV = GridSearchCV(pipeline, parameters, cv=5, n_jobs= -1, verbose=1)
    CV = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    
    CV.fit(Xtrain, Ytrain)
    
    
    print("Best score: %0.3f" % CV.best_score_)
    print("Best parameters set:")
    best_parameters = CV.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
     '''
     
         
    pipeline.fit(Xtrain, Ytrain)
    YpredictionSGD = pipeline.predict(Xtest)    
    print('')
    print(' Accuracy: ', accuracy_score(Ytest, YpredictionSGD))    
    print('')
    
    show_conf_matrix(Ytest, YpredictionSGD)
  
    labels_report=['UI design','development','document organisation','issues','requirements','source code','testing','uncertain','use case']
    print(classification_report(Ytest, YpredictionSGD, target_names=labels_report))
    
    
    get_top_features_for_each_class(pipeline)
  
    second_level_classifier(Xtest, Ytest, YpredictionSGD, Ztest, 'requirements')
    second_level_classifier(Xtest, Ytest, YpredictionSGD, Ztest, 'development')
    
    identify_user_stories(Xtest, Ytest, Ztest)
    
def get_data_statistics():
    X, Y, Z = read_data('CSV Data/snowflake_annotated_data.csv')
 
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



results = get_tags('CSV Data/snowflake_annotated_data.csv', 'programming-language')
get_data_statistics()    
train_classifier()
print(results)
prog_languages = search_words_in_specific_categories('CSV Data/snowflake_annotated_data.csv', 'programming-language')
print('Programming Languages:')
print(prog_languages)


