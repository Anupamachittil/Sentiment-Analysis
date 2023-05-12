import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def trainmodel():
    rew=pd.read_csv("reviews (1).csv")
    rew=rew.rename(columns = {'text': 'review'}, inplace = False)
    X=rew.review
    y=rew.polarity
    #split data
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,random_state=1)
    vector =  CountVectorizer(stop_words='english',lowercase=False)
    #fit the vectorizer on the training data
    vector=CountVectorizer(stop_words="english",lowercase=False)
    vector.fit(X_train)
    X_transformed = vector.transform(X_train)
    X_transformed.toarray()
    X_test_transformed = vector.transform(X_test)
    nb=MultinomialNB()
    nb.fit(X_transformed,y_train)
    saved_model=pickle.dumps(nb)
    s=pickle.loads(saved_model)
    return s,vector
s,vector=trainmodel()

def predict(input):
    vec = vector.transform([input]).toarray()
    pred=(str(list(s.predict(vec))[0]).replace('0','NEGATIVE').replace('1','POSITIVE'))
    return pred

    
st.header("Sentiment Analysis")
input=st.text_input("Enter the review")
if st.button('predict'):
    st.write(predict(input))


    
    


