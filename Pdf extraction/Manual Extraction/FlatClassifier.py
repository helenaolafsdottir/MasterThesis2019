# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:00:19 2019

@author: Lenovo
"""
import numpy as np
import nltk
import itertools
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.pipeline import make_pipeline, Pipeline
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


def read_data(file):
    X = []
    Y = []
    with open(file, encoding='utf-8-sig') as f:
        for line in f:
            data = line.strip().split('|')
         
            X.append(data[0])
            Y.append(data[2])
            
    return X, Y

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
    tr = str.maketrans("","", string.punctuation)
    return sentence

def show_conf_matrix(Ytest, Yprediction):
    cnf_matrix = confusion_matrix(Ytest, Yprediction, labels=['stakeholders', 'non-functional', 'functional','features','other', 'uncertain', 'source code', 'testing','data','document organisation','issues','use case','UI design','APs, components & communication','Technology Solution','Workflows'])
    
    print(cnf_matrix)
    # the ordering of the labels is gotten by running: print('labels: ', pipeline.classes_)
    labels=['stakeholders', 'non-functional', 'functional','features','other', 'uncertain', 'source code', 'testing','data','document organisation','issues','use case','UI design','APs, components & communication','Technology Solution','Workflows']
    
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
    labels=['stakeholders', 'non-functional', 'functional','features','other', 'uncertain', 'source code', 'testing','data','document organisation','issues','use case','UI design','APs, components & communication','Technology Solution','Workflows']
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
   
def train_classifier():
    X, Y = read_data('snowflake_annotated_data.csv')
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=82)
    
    stopWords = set(stopwords.words('english'))
    
    # --------- NLP -------------     
    for pos, sentence in enumerate(Xtrain):
        #sentence = lemma(sentence)
        sentence = stem(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        Xtrain[pos] = sentence
        
    for pos, sentence in enumerate(Xtest):
        #sentence = lemma(sentence)
        sentence = stem(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        Xtest[pos] = sentence

    pipeline = make_pipeline(
        CountVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        TfidfTransformer(norm='l2', sublinear_tf=True),
        
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        #OneVsRestClassifier(LinearSVC(random_state=42, penalty='l2', loss='hinge'))     #67.05
        
        #TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,2)),
        LogisticRegression(n_jobs=1, C=1e3)  #69.36 
        
        #SGDClassifier(alpha=0.0001, loss="hinge", penalty="elasticnet", max_iter=150) #68.80
    
    )    
         
    pipeline.fit(Xtrain, Ytrain)
    YpredictionSGD = pipeline.predict(Xtest)    
    print('')
    print(' Accuracy: ', accuracy_score(Ytest, YpredictionSGD))    
    print('')
    
    show_conf_matrix(Ytest, YpredictionSGD)

    #labels=['stakeholders', 'non-functional requirements', 'functional requirements','feature','other', 'uncertain', 'source code', 'testing','data','document organisation','issues','use cases','interface designs','APs, components and their communication','technology solutions','workflows']
    #print(classification_report(Ytest, YpredictionSGD, target_names=labels_report))
    
    
    #get_top_features_for_each_class(pipeline)
    
def get_data_statistics():
    X, Y = read_data('snowflake_annotated_data.csv')
 
    # the ordering of the labels is gotten by running: print('labels: ', pipeline.classes_)
    labels=['stakeholders', 'non-functional', 'functional','features','other', 'uncertain', 'source code', 'testing','data','document organisation','issues','use case','UI design','APs, components & communication','Technology Solution','Workflows']
    
    data = (Y.count('stakeholders'), Y.count('non-functional'), Y.count('functional'), Y.count('features'), Y.count('other'), Y.count('uncertain'), Y.count('source code'),Y.count('testing'),Y.count('data'),Y.count('document organisation'),Y.count('issues'),Y.count('use case'),Y.count('UI design'),Y.count('APs, components and their communication'),Y.count('Technology solution'),Y.count('workflows'))
    
    #Crate barplot
    pyplot.bar(labels, data)
    pyplot.xticks(rotation='vertical')
    pyplot.title('AK categories')
    pyplot.xlabel('Category')
    pyplot.ylabel('Number of sentences')
    for a,b in zip(labels, data):
        pyplot.text(a, b, str(b))
    pyplot.show()



get_data_statistics()    
train_classifier()