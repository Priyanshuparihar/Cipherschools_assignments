import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
from os import path
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pickle import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')


dataset_loc = "SMSSpamCollection"
image_loc = "spam_img.png"

# Data
def load_data(dataset_loc):
    df=pd.read_csv(dataset_loc,sep='\t',names=['target','message'])
    df['length'] = df['message'].apply(len)
    return df

def load_description(df):

    st.header("EDA (Exploratory Data Analysis) ")
    # Preview of the dataset
    preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
    if(preview == "Top"):
        st.write(df.head())
    if(preview == "Bottom"):
        st.write(df.tail())

    # display the whole dataset
    if(st.checkbox("Show complete Dataset")):
        st.write(df)

    # Show shape
    if(st.checkbox("Display the shape")):
        st.write(df.shape)
        dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
        if(dim == "Rows"):
            st.write("Number of Rows", df.shape[0])
        if(dim == "Columns"):
            st.write("Number of Columns", df.shape[1])

    # show columns
    if(st.checkbox("Show the Columns")):
        st.write(df.columns)

    # show info    
    if(st.checkbox("Show the Data Description")):
        st.write(df.describe(include='all'))
        

def graph(df):
    st.subheader("Graphs of Target column :")
    dim = st.radio("Bar Graph/Pie Chart?", ("Bar Graph", "Pie Chart"))
    if(dim == "Bar Graph"):
        sns.countplot(x="target", data=df)
        plt.xlabel("TARGET")
        plt.ylabel("FREQUENCY")
        plt.title(" BAR PLOT OF : TARGET")
        st.pyplot()
    if(dim == "Pie Chart"):
        plt.title(" PIE CHART OF : TARGET")
        df["target"].value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%1.1f%%', shadow = True)
        plt.ylabel("Spam vs Ham")
        plt.legend(["Ham", "Spam"])
        st.pyplot()
    if(st.checkbox("OBSERVATION 1.1 :")):
        st.write('''
            1. Tagret column have 2 unique vaules (ham and spam).
            2. There are 4825 (86.6%) of ham messages.
            3. There are 747 (13.4%) of spam messages.
            ''')        

    st.subheader("Histogram of 'HAM' and 'SPAM' with respect to Length :")
    df.hist(column='length',by='target', bins=50)
    st.pyplot()
    if(st.checkbox("OBSERVATION 1.2 :")):
        st.write('''
            1. Looks like spam messages are generally longer than ham messages.
            2. Bulk of ham has length below 100, for spam it is above 100.
            3. We will check if this feature is useful for the classification task.
            ''')       


def cleaned(df_sms):

# Join all messages to make one paragraph. 
  words_ = ' '.join(df_sms['message'])
  
# change all data into lower case.
  word_=words_.lower()

#  removes all word like(https,www.)
  c_word = " ".join([word for word in word_.split()
                            if 'http' not in word
                         and 'www.' not in word
                            ])
  
# removes all special characters and digits.
  word_sms=''
  letters_only_sms = re.sub("[^a-zA-Z]", " ",c_word)
 
# removes all stopwords like (the,we,are,it,if......)
  words = letters_only_sms.split()
  words = [w for w in words if not w in stopwords.words("english")]
  
# removes all words which have length less than 2.
  for a in words:
    if len(a)<3:
      words.remove(a)

# again make all words into paragraph.
  for i in words:
    word_sms=word_sms+" "+i

# return that paragraph.
  return word_sms

def word_cloud(df):
    st.subheader("Treating 'SPAM / HAM' messages")
    dim = st.radio("Spam/Ham?", ("Spam", "Ham"))
    if(dim == "Ham"):
        df_ham = df.loc[df['target']=='ham', :]
        st.write(df_ham.head())
        if (path.exists("wc_ham.png")):
            st.image("wc_ham.png", use_column_width = True)
         else:
             cleaned_ham = cleaned(df_ham)

             wordcloud_ham = WordCloud(
                       background_color='black',
                       width=1600,
                       height=800
                      ).generate(cleaned_ham)
             wordcloud_ham.to_file("wc_ham.png")
             st.image("wc_ham.png", use_column_width = True)

    if(dim == "Spam"):
        df_spam = df.loc[df['target']=='spam', :]
        st.write(df_spam.head())
        if (path.exists("wc_spam.png")):
            st.image("wc_spam.png", use_column_width = True)
         else:
             cleaned_spam = cleaned(df_spam)

             wordcloud_spam = WordCloud(
                       background_color='black',
                       width=1600,
                       height=800
                      ).generate(cleaned_spam)
             wordcloud_spam.to_file("wc_spam.png")
             st.image("wc_spam.png", use_column_width = True)


def preprocess(raw_msg):

    stemmer = PorterStemmer()
    # Removing words like (http,www.)
    cleaned = " ".join([word for word in raw_msg.split()
                            if 'http' not in word
                          and 'www.' not in word
                            ])

    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",cleaned)

    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()
    
    # remove stop words                
    words = [w for w in words if not w  in stopwords.words("english")]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    clean_sent = " ".join(words)
    
    return clean_sent
    
def predict(msg):
    
    # Loading pretrained CountVectorizer from pickle file
    vectorizer = load(open('countvectorizer.pkl', 'rb'))
    
    # Loading pretrained logistic classifier from pickle file
    classifier = load(open('logit_model.pkl', 'rb'))
    
    # Preprocessing the tweet
    clean_msg = preprocess(msg)
    
    # Converting text to numerical vector
    clean_msg_encoded = vectorizer.transform([clean_msg])
    
    # Converting sparse matrix to dense matrix
    msg_input = clean_msg_encoded.toarray()
    
    # Prediction
    prediction = classifier.predict(msg_input)
    
    return prediction

def test():
    st.header("Prediction")
    msg = st.text_input('Enter your Message : ')

    prediction = predict(msg)

    if(msg):
        st.subheader("Prediction:")
        if(prediction == 0):
            st.image("spam.jpg", use_column_width = True)
        else:
            st.image("ham.jpg", use_column_width = True)



# Main
def main():

    # sidebar
    #load_sidebar()

    # Title/ text
    st.title('SMS Spam Collection Data Set')
    st.image(image_loc, use_column_width = True)
    st.text("Predict the Message 'SPAM' or 'HAM'.")

    # loading the data
    df = load_data(dataset_loc)

    # display description
    load_description(df)

    graph(df)

    word_cloud(df)
    
    test()

if(__name__ == '__main__'):
    main()