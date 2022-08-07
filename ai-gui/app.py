import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates


import pandas as pd
# from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
import io
#import opendatasets as od
import pandas as pd
import datetime
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
from sklearn.svm import SVR 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
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
        
        
        

#import lstm 

## TO-DO 1 Load data + preprocessing

## TO-DO 2 Train model 
#df_x2 = pd.read_csv("C:\\Users\\pwtan\\Documents\\GitHub\\50.021-AI-Project-2022\\code_folder\\tweet_content_only\\test\\X_test.csv")
#df_y2 = pd.read_csv("C:\\Users\\pwtan\\Documents\\GitHub\\50.021-AI-Project-2022\\code_folder\\tweet_content_only\\test\\y_test.csv")
#df_test = pd.concat([df_x2,df_y2], axis=1)
#X_test = df_x2.iloc[:,0:-1].values

#class CPU_Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else: return super().find_class(module, name)




# import torch.nn as nn
# import torch.nn.functional as F
# class LSTM(torch.nn.Module) :
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout) :
#         super().__init__()

#         # The embedding layer takes the vocab size and the embeddings size as input
#         # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

#         # The LSTM layer takes in the the embedding size and the hidden vector size.
#         # The hidden dimension is up to you to decide, but common values are 32, 64, 128
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

#         # We use dropout before the final layer to improve with regularization
#         self.dropout = nn.Dropout(dropout)

#         # The fully-connected layer takes in the hidden dim of the LSTM and
#         #  outputs a a 3x1 vector of the class scores.
#         self.fc = nn.Linear(hidden_dim, 1)
        
#         self.relu = nn.ReLU()

#     def forward(self, x, hidden):
#         """
#         The forward method takes in the input and the previous hidden state 
#         """

#         # The input is transformed to embeddings by passing it to the embedding layer
#         embs = self.embedding(x)

#         # The embedded inputs are fed to the LSTM alongside the previous hidden state
#         out, hidden = self.lstm(embs, hidden)

#         # Dropout is applied to the output and fed to the FC layer
#         out = self.dropout(out)
#         out = self.fc(out)

#         # We extract the scores for the final hidden state since it is the one that matters.
#         out = out[:, -1]
#         out = self.relu(out)
#         return out.squeeze(), hidden
    
#     def init_hidden(self):
#         return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))


#contents = CPU_Unpickler(open("lstm","rb")).load()

class StreamlitApp:

    def __init__(self):
        with open('final_pickle_ver_6' , 'rb') as f:
            model = pickle.load(f)
        self.model= model
        #self.model = RandomForestClassifier() # replace with our model

   # def train_data(self):  #train our data
   #     self.model.fit(x_train, y_train)
   #     return self.model

    def construct_app(self):


        # df1 = pd.read_csv("Covid-19 Twitter Dataset (Apr-Jun 2020).csv")
        # df2 = pd.read_csv("Covid-19 Twitter Dataset (Aug-Sep 2020).csv")

        # df = pd.concat([df1, df2], ignore_index=True)
        # df = df.reset_index()
        # df.head()
        # df=df.iloc[:,[2, 6, 13, 7]]
        # # convert to ints

        # df["created_at"] = pd.to_datetime(df["created_at"])
        # df['created_at'] = df['created_at'].map(mdates.date2num)
        # df.dropna(inplace=True)


        # x=df.iloc[:,: -1].sample(n=10000).values


        # # Use MinMax scaling on X2 and X3 features
        # scaler=MinMaxScaler()

        # scaler.fit_transform(x)

        # y=scaler.data_range_







        st.title("Retweet Prediction")


        sentiment_class = st.number_input(label="Input Sentiment Class (i.e. 0 for negative, 1 for neutral, 2 for positive)",min_value=0,max_value=2)
        retweet_status= st.number_input(label="Input Retweet Status (i.e. 0 for un-retweeted, 1 for retweeted)", min_value=0,max_value=1)
        day = st.number_input(label="Input Day of Post (i.e. 0-6 for Mon to Sunday)",min_value=0,max_value=6)
        hour = st.number_input(label="Input Hour of Post (i.e. 0-23)",min_value=0,max_value=23)
        #author_class = st.number_input(label="author class",min_value=0)
        #text=len(st.text_area("Enter Tweet Content").split())
        #fav=int(st.text_area("Enter No. of Likes"))

        #scaler=MinMaxScaler()

        #text=st.text_area("Input Tweet Content")
        #date=st.text_area("Input Date")
        #fav=st.number_input("Input Fvaorite Count")
        #dir_sent=st.number_input("Input sentiment value")

        #data={'date':[date]}
        #date_conv=pd.to_datetime(data["date"])
        #date_conv = date_conv.map(mdates.date2num)[0]
        #sia = SentimentIntensityAnalyzer()
        #sent=sia.polarity_scores(text)['compound']

        
        #st.write(sent)

        #x_scale=scaler.fit_transform(np.array([date_conv,int(fav),sent])).reshape(1,-1)
        #st.write(x_scale)
        #h0, c0=self.model.init_hidden()
        #st.write("Predicted no. of retweets:",self.model.predict(np.array([date_conv,int(fav),dir_sent]).reshape(1,-1))) #to run input with the model
        input=np.array([sentiment_class,retweet_status,day,hour])
        prediction=self.model(torch.from_numpy(input.astype(np.float32))).item()
        
        st.write("Output:",int(prediction))
        return self


sa = StreamlitApp()
sa.construct_app()