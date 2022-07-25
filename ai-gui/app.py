import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

## TO-DO 1 Load data + preprocessing

## TO-DO 2 Train model 


class StreamlitApp:

    def __init__(self):
        self.model = RandomForestClassifier() # replace with our model

    def train_data(self):  #train our data
        self.model.fit(x_train, y_train)
        return self.model

    def construct_app(self):
        st.title("Retweet Prediction")

        input=st.text_area("Enter Tweet Content.")
        st.write("Predicted no. of retweets:", input) #to run input with the model

        return self


sa = StreamlitApp()
sa.construct_app()