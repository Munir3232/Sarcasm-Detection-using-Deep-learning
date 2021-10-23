import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump,load

import pandas as pd 
import numpy as np 
import tensorflow as tf 
#import re 
from  tensorflow.keras.layers import Activation ,Input,LSTM,GRU,Dense ,Flatten,Embedding,Dropout,Bidirectional
from tensorflow.keras.utils import plot_model 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences 

#import pickle
from numpy import asarray
from tqdm import tqdm
#from gensim.models import KeyedVectors 
#import pickle

from numpy import zeros
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import tensorflow_hub as hub
from joblib import load ,dump
import warnings
warnings.filterwarnings('ignore')



st.title('Sarcasm Detection')

st.text('Please upload the Data to detect Sarcasm. Data should have following fields(features).')
st.text("['comment', 'author', 'subreddit', 'score', 'ups', 'downs', 'date','created_utc', 'parent_comment']")
data_load_state = st.text('Loading data...')

data = st.file_uploader("SARCASM DATA")
if data is not None:
   df = pd.read_csv(data)
   #df.to_csv('data.csv')
   st.subheader('SARCASM DATA UPLOAD DONE!')
   st.write(df)

data_load_state.text('Loaded data...')

loading_universal_Sentence_encoder=st.text('loading universal sentence encoder')
@st.cache
#Encoder = st.file_uploader("SARCASM DATA")
def load_universal_Sentece_encoder():
	embed = hub.load("5")
	return embed
embed=load_universal_Sentece_encoder()
loading_universal_Sentence_encoder.text('loaded universal sentence encoder')

def SarcasmDetection(df,embed):
  ''' this function takes the data , and universal sentence encoder and returns
  the prediction about sarcasm or not.'''
  
  

  print('\nnull values in comment:-',df['comment'].isna().sum())
  df['comment']=df['comment'].fillna(' ')
  print('after filling null comments:',df['comment'].isna().sum())

  print('null values in parent_comment:-',df['parent_comment'].isna().sum())
  df['parent_comment']=df['parent_comment'].fillna(' ')
  print('after filling null parent_comment:',df['parent_comment'].isna().sum())
  
  

  ##Response Encoding of categorical features
  # fit and transform functions for response encoding of categorical variable
  def fit_response_encoding(df,feature):
    ''' this function takes df, feature and 
        return dict of preobabilities of categories in feature'''
    # we need class info for response encoding so add class feature to df
    
    CountOfFeatures_given_yes=df[df['label']==1][feature].value_counts()
    CountOfFeatures=df[feature].value_counts()
    # if any Category is present in CountOfFeatures and not in CountOfFeatures_given_yes then for that Category
    # its value in dict_of_probability will be nan and we can replace this with 0 as it is in CountOfFeatures
    # due fact that it has only non surcastic comments so its probability of surcastic comment will be zero 
    
    dict_of_probability=(CountOfFeatures_given_yes/CountOfFeatures)
    dict_of_probability=dict_of_probability.fillna(0).to_dict()
    return dict_of_probability


  def transform_response_encoding(df,feature,dict_fit):
      ''' this function takes the df,feature for reponse coding  and dict learned by fit function 
        and returns df with new feature_response feature with response coding of feature'''
      
      #keys_of_feature_to_transform=df[feature].value_counts().to_dict().keys()
      df[feature+'_response']=df[feature]
      df[feature+'_response']=df[feature+'_response'].map(dict_fit)
      # laplace smooting, if category is not present in train then it's probability will be 0.5 
      df[feature+'_response']=df[feature+'_response'].fillna(0.5) 
      
      return  df


  

  #response coding of categorical features
  response_DictSubreddit=load('response_DictSubreddit.joblib')
  response_DictAuthor=load('response_DictAuthor.joblib')

  
  df=transform_response_encoding(df,'author',response_DictAuthor)

  df=transform_response_encoding(df,'subreddit',response_DictAuthor)
  print('\nResponse coding is Done!')

  #scaling numerical features
  scaler=load('scaler.joblib')
  scaled_train=scaler.transform(df[['score','ups','downs']].values)
  df[['scoreScaled','upsScaled','downsScaled']]=scaled_train
  
  print('\nScaling of Numerical features is Done!')
  print('loading Universal Sentence Encoder...')
  # using Universal Sentence Encoder
  def embedding_using_universalSentenceEncoder(x_train,feat,embed,chunk_size=1):
      ''' this function takes dataframe ,feature name,universal sentence encoder 
      and creates the embedding of points in chunks and if any porblem comes while 
      processing chunk of points then processes one point at a time and then lastly 
      processes remaing points .'''
      
      start=np.arange(0,len(x_train),chunk_size)[0]  # start of set , this will be updated after processing of each set , end point of last set becomes start of next set
      t1=embed([x_train[feat].iloc[0]])             # creating one embedding to add the set of embeddings to it to create matrix of embeddngs
      for steps in tqdm(np.arange(0,len(x_train),chunk_size)[1:]):
        ### process in chunks
        try:
          t2=embed(x_train[feat].iloc[start:steps]) # embedding a set of points at a time to save time
          start=steps                                    # end point of last set becomes start of next set
          t1=tf.concat([t1, t2], 0)                      # adding embeddins to main matrix of embeddings
          #print('start:',start,'steps:',steps)

        ###process single point
        except:    # if any problem occurs while processing (embedding)  set of points then embed them using one point at a time 
          for OneStep in range(start,steps):             # iterating one step
            t2=embed([x_train[feat].iloc[OneStep]])
            t1=tf.concat([t1, t2], 0)
          print('start of chunk:',start,'end of chunk:',steps)
          start=steps                                     #end point of last set becomes start of next set

      ###process remaining points after processing in chunks
      # now for raming points
      print('last step end:',steps)
      if steps==len(x_train):
        print(True)
      else:
        print('remaining points:',len(x_train)-steps)
        for remainingSteps in tqdm(range(steps,len(x_train))):
          t2=embed([x_train[feat].iloc[remainingSteps]])
          t1=tf.concat([t1, t2], 0)
        print('last point:',remainingSteps)

      return t1

  
 

  #embedding of comment
  print('\nEmbedding of comment is bing processed...')
  comment_embeddings=embedding_using_universalSentenceEncoder(df,'comment',embed)

  #embdedding of paret_comment
  print('\nEmbedding of parent_comment is bing processed...')
  parent_comment_embeddings=embedding_using_universalSentenceEncoder(df,'parent_comment',embed,chunk_size=1)


  print('\nEmbedding using universal sentence encoder is done!')

  
  #creating a variable that is numpy array to use in train dataset creation
  NumericalFeaturesTrain=df[['scoreScaled','upsScaled','downsScaled','author_response','subreddit_response']].values

  #Creating Dataset  
  dataset = tf.data.Dataset.from_tensor_slices(({"InputComment":comment_embeddings[1:],  # here i  created one(1st) embedding to add the set of embeddings to it to create matrix of embeddngs
                                                "InputParentComment":parent_comment_embeddings[1:],
                                                "inputNumerical":NumericalFeaturesTrain}))
  dataset = dataset.shuffle(2)
  dataset = dataset.batch(10, drop_remainder=False)
  dataset=dataset.prefetch(tf.data.AUTOTUNE)

  print('\nCreated Dataset!')
  #loading model
  model = tf.keras.models.load_model('UniversalSentenceEncoderbest_model_13_8_2021SGD1.hdf5')
  print('\nloaded model')

  print('\npredicting...')
  # predictions
  return [np.argmax(i) for i in model.predict(dataset)]  



if st.checkbox('uploaded all files '):

   predictions=SarcasmDetection(df,embed)
   print('Prediction is Done!')
   
   dict1={1:'Sarcastic',0:'Non Sarcastic'}
   labels=[]
   for i in predictions:
      labels.append(dict1[i])
   st.subheader('Labels:-')
   st.write(labels)
   st.subheader('Prediction Done!')