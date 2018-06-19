import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnnA import TextCNNRNN
from sklearn.model_selection import train_test_split
try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
    from win_unicode_console import enable
    enable()
except ImportError:
    pass
logging.getLogger().setLevel(logging.INFO)

#extraire les dictionnaires entre en parameters la liste des donnees
def dictionnaire_extration(data):
    words = []
    for corpus_raw in data:
        #on converti en minuscule
        corpus_raw = corpus_raw.lower()
        #on extrait tout les mots
        for word in corpus_raw.split():
            if word!='.':
                words.append(word)
    #on supprime les mots duplique
    words = set(words)
    #les  dictionnaire
    word2int = {}
    int2word = {}
    #on remplis les dictionnaires
    vocab_size = len(words)
    for i,word in enumerate(words):
        word2int[word] = i
        int2word[i] = word
    return word2int,int2word,vocab_size




# maintenant qu on a les donnees , on doit la representer d'une façon a que le pc la comprene cv dire les nombre
def to_one_hot(data_point_index,vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def entrainement_extraction(data_X,word2int):
    sentences = []
    for corpus_raw in data_X:
        raw_sentences = corpus_raw.split('.')
        for sentence in raw_sentences:
            sentences.append(sentence.split())

    WINDOW_SIZE = 2
    data = []
    for sentence in sentences:
        for word_index,word in enumerate(sentence):
            for nb_word in sentence[max(word_index - WINDOW_SIZE,0):min(word_index + WINDOW_SIZE,len(sentence)) + 1]:
                if nb_word!=word:
                    data.append([word,nb_word])
    x_train = [] #le mot d'entree
    y_train = [] #le mot de sortie

    for data_word in data:
        x_train.append(to_one_hot( word2int[ data_word[0] ] , vocab_size ))
        y_train.append(to_one_hot( word2int[ data_word[1] ] , vocab_size ))

    # on a les converture en numpy array
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train,y_train



x_ = data_helper.load_data_phrase('data/train1.csv.zip')
word2int,int2word,vocab_size = dictionnaire_extration(x_)
print(vocab_size)
x_train,y_train = entrainement_extraction(x_,word2int)






# CONTRUIRE LE MODEL TENSORFLOW

x = tf.placeholder(tf.float32,shape=[None,vocab_size])
y_label = tf.placeholder(tf.float32,shape=[None,vocab_size])


# premiere etape en prend nos donnes et les convertie en une embedded data
EMBEDDING_DIM = 5

W1 = tf.Variable(tf.random_normal([vocab_size,EMBEDDING_DIM]))
b1  = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #le bias
hidden_representation = tf.add(tf.matmul(x,W1),b1)


#Apres en ce sert de dim embedded pour faire les prediction on utilisant softmax

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM,vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))

#utilisation du softmax
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation,W2),b2))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #on initialize les variables

#on definit la fonction de perte
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(prediction),reduction_indices=[1] ) )

#on definit le pas d'entrainement
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iter = 10000




# on commence l'entrainement :

for _ in range(n_iter):
    sess.run(train_step,{x : x_train,y_label : y_train})
    print('loss : ',sess.run(cross_entropy_loss,{x : x_train,y_label : y_train}))


emb1 = W1[1]
emb = tf.add(emb1,b1)

print(sess.run(W1))
print('------------------------------')
print(sess.run(b1))
print(sess.run(emb1))
print(sess.run(emb))
