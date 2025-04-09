#Deployment Process Code
#importing libraries
import streamlit as st
from joblib import load
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#function for pre-process the text
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(lemmatizer.lemmatize(i))

  return " ".join(y)

#loading the downloaded files
Tfidf = load('vectorizerSVM.joblib')
model = load('modelSVM.joblib')

#streamlit titles
st.title("Data Science Project by Nij")
st.title("Email/SMS Spam Classifier")

#creating box for message input
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = Tfidf.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
