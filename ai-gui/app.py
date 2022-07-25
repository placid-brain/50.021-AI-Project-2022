import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# iris_data = load_iris()
# # separate the data into features and target
# features = pd.DataFrame(
#     iris_data.data, columns=iris_data.feature_names
# )
# target = pd.Series(iris_data.target)

# # split the data into train and test
# x_train, x_test, y_train, y_test = train_test_split(
#     features, target, test_size=0.2, stratify=target
# )

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