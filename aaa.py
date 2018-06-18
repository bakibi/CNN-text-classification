import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf


try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
    from win_unicode_console import enable
    enable()
except ImportError:
    pass
logging.getLogger().setLevel(logging.INFO)

size_xxxx = 33

dict = {"دوريات":1,"أبطال ":3,"le":5,"الأوروبي ":7,"مرحبا" : 11}
a =  max(dict, key=lambda i: dict[i])
nv = np.chararray((dict[a]+1),itemsize=30,unicode=True)
nv[:] = ''
for a in dict.keys():
	nv[dict[a]] =a





#input in case of prediction
string_to_manipulate = tf.placeholder(tf.string , [None])
#input in training phase
input = tf.placeholder(tf.int32 , [None])
#where our dictionnaire of value is stored
dictionnaire = tf.constant(nv)
#we split the description into list of string values

string_to_manipulate1 = string_to_manipulate[0]
splited_string = tf.string_split([string_to_manipulate1],delimiter=' ',skip_empty=True)
splited_string_values = splited_string.values

mapped_list_of_string = tf.map_fn(
	lambda x: tf.cond(
		tf.equal(
			tf.size(
				tf.where(
					tf.equal(dictionnaire, [x])
					)[::,-1]
				)
				,0),
					lambda :tf.constant(0,dtype=tf.int32), # si oui
						lambda : tf.cast(
							tf.where(
								tf.equal(dictionnaire, [x])
										)[::,-1][0],
										dtype=tf.int32
										)
									),splited_string_values,dtype=tf.int32)
r = tf.cond(
	tf.less(
		size_xxxx-tf.size(splited_string)
		, 0
		)
			,lambda : 0 , #yes
			 lambda  :size_xxxx-tf.size(splited_string) #no
			 )
extra_zero = tf.zeros([r],dtype=tf.int32 )
ccc  = tf.cond(tf.less(size_xxxx-tf.size(splited_string), 0),lambda : 0 , lambda  :size_xxxx-tf.size(splited_string))
input  =tf.concat([ mapped_list_of_string , tf.zeros([ccc],dtype=tf.int32 ) ],0 )
input = tf.cond( tf.less(size_xxxx, tf.size( input )),lambda : tf.slice( input ,[0],[size_xxxx]), lambda  :input )

#for multiple input

final_input = tf.map_fn(lambda x:tf.string_split([x],delimiter=' ',skip_empty=True).values,string_to_manipulate)
final_input = tf.map_fn(lambda x1:tf.map_fn(lambda x: tf.cond(tf.equal(tf.size(tf.where(tf.equal(dictionnaire, [x]))[::,-1]),0),lambda :tf.constant(0,dtype=tf.int32),lambda :tf.cast(tf.where(tf.equal(dictionnaire, [x]))[::,-1][0],dtype=tf.int32)),x1,dtype=tf.int32),final_input,dtype=tf.int32)
final_input = tf.map_fn(lambda solv:tf.concat([ solv , tf.zeros([tf.cond(tf.less(size_xxxx-tf.size(solv), 0),lambda : 0 , lambda  :size_xxxx-tf.size(solv))],dtype=tf.int32 ) ],0 ),final_input)
final_input = tf.map_fn(lambda solv:tf.cond( tf.less(size_xxxx, tf.size( solv )),lambda : tf.slice( solv ,[0],[size_xxxx]), lambda  :solv ),final_input)



stri  =tf.constant("تنسيق أمني مغربي إسباني يطيح بخلية إرهابية موالية")
stri1 = tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(tf.regex_replace(stri,"\s{2,}", " "),"\?", " \? "),"\)", " \) "),"\(", " \( "),"!", " ! "),",", " , "
	),"\'ll", " \'ll"),"\'d", " \'d"),"\'re", " \'re"),"n\'t", " n\'t"),"\'ve", " \'ve"),"\'s", " \'s")," : ", ":"),"[^\u0627-\u064aA-Za-z0-9:(),!?\'\`]", " ")

#verifie si un nom existe dans la liste dictionnaire et retourne sans indoce si il existe et retourne 0 sinon
#a = tf.cond(tf.equal(tf.size(tf.where(tf.equal(dictionnaire, [x]))[::,-1]),0),lambda :tf.constant(0,dtype=tf.int64),lambda :tf.where(tf.equal(dictionnaire, [x]))[::,-1][0])

with tf.Session() as sess:
	print(sess.run(stri1,{string_to_manipulate : ["salut tout le monde ","salut tout le monde "]}))
