import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import nltk

# Download the lexicon
nltk.download("vader_lexicon")

import pandas as pd
# from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
import io
#import opendatasets as od
import pandas as pd
import datetime
import matplotlib.dates as mdates
import torch.nn as nn
import torch.nn.functional as F

class DNNRegressor(nn.Module):
    def __init__(self, n_input_features):
        super(DNNRegressor, self).__init__()
        
        self.linear1 = nn.Linear(n_input_features, 50)
        
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(50,10)
        
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(10,1)
        
        
        
    def forward(self,x):
        
        linear1 = self.linear1(x)
        
        relu1= self.relu1(linear1)
        
        linear2 = self.linear2(relu1)
        
        relu2 = self.relu2(linear2)
        
        linear3 = self.linear3(relu2)
        
        
        return linear3
        

class StreamlitApp:

    def __init__(self):
        with open('final_pickle_ver_7' , 'rb') as f: #load DNNRegressor trained model
            model = pickle.load(f)
        self.model= model

    def construct_app(self):

        st.title("Retweet Prediction")

        text=st.text_area(label="Input Tweet Content (max. 280 chars)",max_chars=280)
        sia = SentimentIntensityAnalyzer()
        sentiment_class=sia.polarity_scores(text)['compound']
        st.write("Vader Sentiment Compound:",sentiment_class)

        #sentiment_class = st.number_input(label="Input Sentiment Class (i.e. 0 for negative, 1 for neutral, 2 for positive)",min_value=0,max_value=2)
        retweet_status= st.number_input(label="Input Retweet Status (i.e. 0 for un-retweeted, 1 for retweeted)", min_value=0,max_value=1)
        day = st.number_input(label="Input Day of Post (i.e. 0-6 for Mon to Sun)",min_value=0,max_value=6)
        hour = st.number_input(label="Input Hour of Post (i.e. 0-23)",min_value=0,max_value=23)
        #author_class = st.number_input(label="author class",min_value=0)
        
        input=np.array([sentiment_class,retweet_status,day,hour])
        prediction=self.model(torch.from_numpy(input.astype(np.float32))).item()
        st.metric(label="**Predicted No. of Retweets**",value=int(prediction))
        return self


sa = StreamlitApp()
sa.construct_app()