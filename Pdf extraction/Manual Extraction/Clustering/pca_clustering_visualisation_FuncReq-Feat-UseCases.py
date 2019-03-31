# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:16:52 2019

@author: Helena Olafsdottir
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:00:19 2019

@author: Lenovo
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
from sklearn.mixture import GaussianMixture
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

    
    print(len(Xreq))
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
    Xi, Y, Z = get_requirements_data()
    
    stopWords = set(stopwords.words('english'))
    
    # --------- NLP -------------     
    for pos, sentence in enumerate(Xi):
        #sentence = lemma(sentence)
        sentence = remove_digits(sentence)
        sentence = remove_punctuations(sentence)
        sentence = remove_digits(sentence)
        Xi[pos] = sentence
        
    
    
    Z_1 = [1,1,1,1,1,1,2,2,2,2,2,0,0,0,0,0,0,0,1,1,2,0,0,0,0,0,1,1,2,1,1,2,2,2,2,0,0,1,1,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,2,2,0,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Z_2 = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 2, 2, 2, 2, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Z_3 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Z_4 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    Z_5 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    
    
    true_k = 3
    random_state=649
    
    pipeline = make_pipeline(
         TfidfVectorizer(stop_words=stopWords, max_features=5500, analyzer='word', lowercase=True, ngram_range=(1,1)),
    )
    X = pipeline.fit_transform(Xi).todense()
    
    
    ## PCA ##
    
    pca = PCA(n_components=2, random_state=200).fit(X)
    data2D = pca.transform(X)   
    
    
    
    ## CLUSTERING ##
    
    kmFitted = KMeans(n_clusters=true_k, random_state=random_state).fit(data2D)
    km = KMeans(n_clusters=true_k, random_state=random_state).fit_predict(data2D)
    gm = GaussianMixture(n_components=true_k, random_state=random_state).fit(data2D).predict(data2D)
        
    
    
    ## EVALUATION ##
    
    mistake_km_0 = []
    mistake_km_1 = []
    mistake_km_2 = []
    mistake_gm_0 = []
    mistake_gm_1 = []
    mistake_gm_2 = []
    
    classification_colors_gm = ['None'] * len(Xi)
    classification_colors_km = ['None'] * len(Xi)
    for i, label in enumerate(gm):
        
        true_label_km = Z_5[i]
        pred_label_km = kmFitted.labels_[i]
        if true_label_km == 0:
            if(pred_label_km != 0): 
                mistake_km_0.append(Xi[i])
                classification_colors_km[i] = 'red'
        if true_label_km == 1:
            if(pred_label_km != 1): 
                mistake_km_1.append(Xi[i])
                classification_colors_km[i] = 'red'
        if true_label_km == 2:
            if(pred_label_km != 2): 
                mistake_km_2.append(Xi[i])
                classification_colors_km[i] = 'red'
        
        true_label_gm = Z_3[i]
        pred_label_gm = gm[i]
        if true_label_gm == 0: 
            if(pred_label_gm != 0): 
                mistake_gm_0.append(Xi[i])
                classification_colors_gm[i] = 'red'
        if true_label_gm == 1:
            if(pred_label_gm != 1): 
                mistake_gm_1.append(Xi[i])
                classification_colors_gm[i] = 'red'
        if true_label_gm == 2:
            if(pred_label_gm != 2): 
                mistake_gm_2.append(Xi[i])
                classification_colors_gm[i] = 'red'
        
    mistakes_km = mistake_km_0 + mistake_km_1 + mistake_km_2
    mistakes_gm = mistake_gm_0 + mistake_gm_1 + mistake_gm_2
    
    numMistakes_km = len(mistakes_km)
    numMistakes_gm = len(mistakes_gm)
    numDatapoints = len(Xi)
    
    print('')
    print('------ KM Accuracy ------')
    print('Mistakes: ', numMistakes_km)
    print('Datapoints: ', numDatapoints)
    print('Accuracy: ', (numDatapoints-numMistakes_km)/numDatapoints)
    
    print('Accuracy class 0 - Purchase Products: ', (Z_5.count(0)-len(mistake_km_0))/Z_5.count(0))
    print('Accuracy class 1 - Browse Products: ', (Z_5.count(1)-len(mistake_km_1))/Z_5.count(1))
    print('Accuracy class 2 - User Management: ', (Z_5.count(2)-len(mistake_km_2))/Z_5.count(2))
    
    print('')
    print('------ GM Accuracy ------')
    print('Mistakes: ', numMistakes_gm)
    print('Datapoints: ', numDatapoints)
    print('Accuracy: ', (numDatapoints-numMistakes_gm)/numDatapoints)
    
    print('Accuracy class 0 - Purchase Products: ', (Z_3.count(0)-len(mistake_gm_0))/Z_3.count(0))
    print('Accuracy class 1 - Browse Products: ', (Z_3.count(1)-len(mistake_gm_1))/Z_3.count(1))
    print('Accuracy class 2 - User Management: ', (Z_3.count(2)-len(mistake_gm_2))/Z_3.count(2))



    
    
    ## VISUALISATION ##
    
    #True values 
    figure(num=None, figsize=(8, 6), dpi=80)
    figure(figsize=(30,30))
    plt.scatter(data2D[:,0], data2D[:,1], c=Z_4, s=200)
    
    
    #Predicted values
    plt.figure(figsize=(35,15))
    plt.subplot(121, title='K-means')
    plt.scatter(data2D[:,0], data2D[:,1], c=km, s=130, edgecolors=classification_colors_km) 
    plt.subplot(122, title='Gaussian Mixture')
    plt.scatter(data2D[:,0], data2D[:,1], c=gm, s=130, edgecolors=classification_colors_gm)
    
    


train_classifier()



