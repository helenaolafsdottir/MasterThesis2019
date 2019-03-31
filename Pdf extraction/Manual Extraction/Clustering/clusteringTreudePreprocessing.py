# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:00:19 2019

@author: Lenovo
"""
import numpy as np
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans



def get_requirements_data():
    X, Y, Z, treude = read_data('../CSV Data/snowflake_annotated_data.csv')
    
    reqs = []
    for i, label in enumerate(Z):
        if(label == 'functional' or label == 'use case' or label == 'features'):
            reqs.append(i) 
    
    Xreq = []
    Yreq = []
    Zreq = []
    Treq = []
    for value in reqs:
        Xreq.append(X[value])
        Yreq.append(Y[value])
        Zreq.append(Z[value])
        Treq.append(treude[value])

    return Xreq, Yreq, Zreq, Treq
    

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
    
    
    sentences, treude_results = read_treude()            
    
    return X, Y, Z, treude_results

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


def remove_stop_words(sentence):
    stopWords = stopwords.words('english')
    token_words = word_tokenize(sentence)
    
    stop_words_removed = []
    
    for word in token_words:
        if word not in stopWords:
            stop_words_removed.append(word)
            stop_words_removed.append(" ")

    return "".join(stop_words_removed)




def read_treude():
    
    file = 'treude_results.csv'
      

    with open(file, encoding='utf-8-sig') as f:
        sentences = []
        treude_sentences = []

        for j, line in enumerate(f):
            
            if (j % 2 != 0):

                treude_sentence = []
                
                data = line.split(']}{[')
                
                for line in data:
                    line = line.strip('{[]}\n').split('] [')
                    for word in line:
                        
                        #do not add sentences that don't contain any words
                        if not word or word == ' ':
                            line.remove(word)
                    
                    treude_sentence.append(line)
                             
                treude_sentences.append(treude_sentence)
             
            else:
                sentences.append(line)
                
    return sentences, treude_sentences





    
def train_classifier():

    X, Y, Z, treude_res = get_requirements_data()

    for truede in treude_res:
       for i, res in enumerate(truede):
           res = ' '.join(res)
           truede[i] = res    
       
    for i, tr in enumerate(treude_res):
        tr = ' '.join(tr)
        treude_res[i] = tr
        
    
#    print(X[1:10])
#    print('')
#    print(Y[1:10])
#    print('')
#    print(Z[1:10])
#    print('')
#    print(treude_res[1:10])
    
        
       
    
    stopWords = set(stopwords.words('english'))
    
    # --------- NLP -------------     
    for pos, sentence in enumerate(X):
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        X[pos] = sentence
  
    true_k = 3

    pipeline = make_pipeline(
        TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,1)),
        KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        
    )
    pipeline.fit(treude_res)
    kmeans = pipeline.named_steps['kmeans']
    tfidf = pipeline.named_steps['tfidfvectorizer']
    
    for i, label in enumerate(kmeans.labels_):
        print(X[i])
        print(treude_res[i])
        print('Label:', label)
    
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    print('')
    print('')
    for i in range(true_k):
        print(i)
        for ind in order_centroids[i, :10]:
            print(terms[ind])
        print('')
    


train_classifier()


