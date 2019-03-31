# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:00:19 2019

@author: Helena Olafsdottir
"""
import numpy as np
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer , LancasterStemmer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import ExtraTreesClassifier
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sqlite3 import Error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, Doc2Vec


def get_requirements_data():
    X, Y, Z = read_data('../CSV Data/snowflake_annotated_data.csv')
    
    reqs = []
    for i, label in enumerate(Z):
        if(label == 'functional' or label == 'use case' or label == 'features'):
        #if((label == 'functional' and "As a" in X[i]) or label == 'features') :
            reqs.append(i) 
    
    Xreq = []
    Yreq = []
    Zreq = []
    for value in reqs:
        Xreq.append(X[value])
        Yreq.append(Y[value])
        Zreq.append(Z[value])

    return Xreq, Yreq, Zreq
    

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


def remove_stop_words(sentence):
    stopWords = stopwords.words('english')
    token_words = word_tokenize(sentence)
    
    stop_words_removed = []
    
    for word in token_words:
        if word not in stopWords:
            stop_words_removed.append(word)
            stop_words_removed.append(" ")

    return "".join(stop_words_removed)




    
def train_classifier():
    X, Y, Z = get_requirements_data()
    stopWords = set(stopwords.words('english'))
    
    # --------- NLP -------------     
    for pos, sentence in enumerate(X):
        #sentence = lemma(sentence)
        sentence = remove_digits(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        X[pos] = sentence


    true_k = 3
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,1)),
        KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    )
    
    pipeline.fit(X)
    kmeans = pipeline.named_steps['kmeans']
    tfidf = pipeline.named_steps['tfidfvectorizer']
    
    for i, label in enumerate(kmeans.labels_):
        print(X[i])
        print('Label:', label)
    
    
    #Zz_1 = [1,1,1,1,1,1,2,2,2,2,2,0,0,0,0,0,0,0,1,1,2,0,0,0,0,0,1,1,2,1,1,2,2,2,2,0,0,1,1,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,2,2,0,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Zz_2 = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 2, 2, 2, 2, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
       
    wrongly_classified = []
    for i, label in enumerate(kmeans.labels_):
        print(X[i])
        print('True label: ', Zz_2[i])
        print('Label:', label)
        
        if Zz_2[i] != label:
            wrongly_classified.append(X[i])


    print('')
    print('')
    print('')
    print("Top terms per cluster:")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print('')

    #model = gensim.models.Word2Vec(X, size=100)
    #w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    

    ## Selecting the best K - SILHOUETTE SCORE ##
    ## Code from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    range_n_clusters = [2, 3, 4, 5, 6]
    X = tfidf.fit_transform(X)
    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=200)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
    
        #sample_silhouette_values = silhouette_samples(Xx, cluster_labels)
    
        
#    for i, z in enumerate(Zz_1):
#        if z == 0: Zz_1[i] = 2
#        if z == 1: Zz_1[i] = 0
#        if z == 2: Zz_1[i] = 1
#    print(Zz_1)
    
    

train_classifier()




#X, Y, Z = get_requirements_data()
#
#for pos, sentence in enumerate(X):
#    #sentence = lemma(sentence)
#    sentence = stem(sentence)
#    sentence = remove_punctuations(sentence)
#    sentence = remove_digits(sentence)
#    sentence = remove_stop_words(sentence)
#    X[pos] = sentence
#
#
#sentences = []
#for sentence in X:
#    token_sentence = word_tokenize(sentence)
#    sentences.append(token_sentence)
#       
# 
#
#model = Word2Vec(sentences, min_count=1)
#
#
##print (model.similarity('this', 'is'))
##print (model['the'])
##print (model.most_similar(positive=['purchase'], negative=[], topn=10))
#print(model.most_similar('page'))
#print('------------------------')
#
#X = model[model.wv.vocab]
#
#kclusterer = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)
#assigned_clusters = kclusterer.fit(X)
##print('------------------------')
##print(assigned_clusters.labels_)
##print('------------------------')
#
##words = list(model.wv.vocab)
##for i, word in enumerate(words):  
##    print(word + ":" + str(assigned_clusters.labels_[i]))
#    
#tsne_plot(model)