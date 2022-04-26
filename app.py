from flask import Flask,render_template,request
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from pathlib import Path
import string
import re
import joblib
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer=WordNetLemmatizer()
nltk.download('wordnet')

def get_text(msg):
    input_text=msg
    input_text=[input_text]
    df_input=pd.DataFrame(input_text,columns=['questions'])
    df_input
    return df_input

model=load_model(r'C:\Users\HP\OneDrive\Desktop\chatbot\model-v1.h5')
tokenizer_t=joblib.load(r'C:\Users\HP\OneDrive\Desktop\chatbot\tokenizer_t.pkl')
vocab=joblib.load(r'C:\Users\HP\OneDrive\Desktop\chatbot\vocab.pkl')

def tokenizer(entry): 
    tokens = entry.split() 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    tokens = [re_punc.sub('', w) for w in tokens] 
    tokens = [word for word in tokens if word.isalpha()] 
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens] 
    tokens = [word.lower() for word in tokens if len(word) > 1] 
    return tokens

def remove_stop_words_for_input(tokenizer,df,feature): 
    doc_without_stopwords = [] 
    entry = df[feature][0] 
    tokens = tokenizer(entry) 
    doc_without_stopwords.append(' '.join(tokens)) 
    df[feature] = doc_without_stopwords
    return df

def encode_input_text(tokenizer_t,df,feature): 
    t = tokenizer_t
    entry = entry = [df[feature][0]] 
    encoded = t.texts_to_sequences(entry) 
    padded = pad_sequences(encoded, maxlen=16, padding='post') 
    return padded

def get_pred(model,encoded_input): 
    pred = np.argmax(model.predict(encoded_input)) 
    return pred

def bot_precausion(df_input,pred): 
    words = df_input.questions[0].split() 
    if len([w for w in words if w in vocab])==0 : 
        pred = 1
    return pred

def get_response(df2,pred): 
    upper_bound = df2.groupby('labels').get_group(pred).shape[0] 
    r = np.random.randint(0,upper_bound) 
    responses = list(df2.groupby('labels').get_group(pred).response) 
    return responses[r] 


app = Flask(__name__) 

@app.route("/") 
def home(): 
    return render_template("index.html",template_folder='templates') 

@app.route("/get") 
def get_bot_response(): 
    userText = request.args.get('msg') 
    df_input = get_text(userText) 
    df2 = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\chatbot\response.csv") 
 #load artifacts
    tokenizer_t = joblib.load(r'C:\Users\HP\OneDrive\Desktop\chatbot\tokenizer_t.pkl') 
    vocab = joblib.load(r'C:\Users\HP\OneDrive\Desktop\chatbot\vocab.pkl') 
    df_input = remove_stop_words_for_input(tokenizer,df_input,'questions') 
    encoded_input = encode_input_text(tokenizer_t,df_input,'questions') 
    pred = get_pred(model,encoded_input) 
    pred = bot_precausion(df_input,pred) 
    response = get_response(df2,pred) 
    return str(response) 


if __name__ == "__main__": 
    app.run() 
    