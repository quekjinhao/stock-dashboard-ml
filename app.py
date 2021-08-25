import dash
import pandas_datareader.data as web
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas_datareader.data as web
#from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
import pylab

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
import yfinance as yf
from pmdarima import auto_arima
from datetime import datetime
import chart_studio.plotly as py
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf

from sklearn.preprocessing import MinMaxScaler
import random

import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


app = dash.Dash()
server = app.server

app.layout = html.Div([
                    
                    # Banner
                    html.H1("Stock LSTM Prediction", style={"text-align":"center", "background-color":"Black", "color":"White", "font-size":60}),
    
                    
                    # Stock picker
                    html.Div([html.H3("Enter Stock Ticker",
                                      style={"text-align":"center", "color":"White"}),
    
                               dcc.Input(id="stockpicker", value="TSLA",
                                         style={"fontSize":24,"width":100, "text-align":"center"})
                             ],
                            style={"display":"inline-block", "verticalAlign":"top" , "width":"45vw","textAlign": "center", "border":"2px solid black"}
                            ),
    
                    
    
                    # Input Picker
                    html.Div([html.H3("Select Date Range", style={"color":"White"}),
                             dcc.DatePickerRange(id="datepicker", min_date_allowed=datetime(1970,1,1),
                                                                    max_date_allowed=datetime.today(),
                                                                    start_date="1970-01-02",
                                                                    end_date =  datetime.today().strftime('%Y-%m-%d'),
                                                                    display_format="D-M-Y")
                             ],
                            style={"display":"inline-block","verticalAlign":"top", "textAlign":"center", "width":"45vw", "border":"2px solid black"}
                            ),
                    
                    
    
                    # sUBMIT and train buttons
                    html.Div([
                        
                                html.Div([html.Button(id="submit_button", n_clicks=0, children="Submit", 
                                    style={"fontSize": 24, "width":"10vw" })],
                                        
                                        style={"text-align":"center", "padding-bottom":10}),
                                        
                              
                                html.Div([html.Button(id="train_button", n_clicks=0, children="Train", 
                                      style={ "fontSize": 24, "width": "10vw"})],
                                        
                                        style={"text-align":"center","padding-bottom":10})
                    
                    ],
                            
                            
                            style={"backgroundColor":"Black", "border-bottom":"2px solid White",
                                   "font-family":"FreeMono, monospace", "font-size":20, "font-color":"Black"}),
                            
                            
                    
    
                    # Main Graph
                    dcc.Loading(id="Loading", children = dcc.Graph(id="graph", style={'width': '100vw', 'height': '85vh'}),
                                type="circle", style={"Size":"30", "Color":"Pink"}),
    
    
                    
                            
                            
],style = {"backgroundColor":"Black"}) 




@app.callback(Output("graph", "figure"),
              [Input("submit_button", "n_clicks")],
              [Input('train_button', 'n_clicks')],
              
             [State("stockpicker", "value"),
              State("datepicker","start_date"), 
              State("datepicker","end_date")])



def update_dashboard(submit_button_click, train_button_click, stock_picked, start_date, end_date):
    global df
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        button_id = "submit_button"
    else: 
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    
    if (button_id=="submit_button"):
        try:
            df = yf.download(stock_picked, start=start_date, end=end_date, verbose=False)
        except:
            pass
        else:
            fig = px.line(df, x=df.index, y=df["Close"], title=stock_picked)
            fig.update_layout(title_x=0.5, hovermode="x", showlegend=True, plot_bgcolor="White", paper_bgcolor="Black",
                             font=dict(family="Courier New, monospace", size=20, color="White"))
            
            return fig
    
    elif (button_id=="train_button"):
        df1 = pd.DataFrame(df["Close"] )

        train_dates = df1.index

        df1 = df1.astype(float)
        #df1 = df1.asfreq("B")

        training_size = int(len(df1)*0.8)
        test_size = (len(df1)-training_size)
        
        train_data = df1.iloc[0:training_size, :]
        test_data = df1.iloc[training_size:len(df), :]
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(train_data)
        
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        
        # n_samples  x window x n_features


        # rolling window 
        n_future = 1
        n_past = 5
        
        
        def split_data(df, n_past, n_future):
            dataX = []
            dataY = []
    
    
            for i in range(n_past, len(df) - n_future +1):
                dataX.append(df[i - n_past:i, 0:df.shape[1]])
                dataY.append(df[i + n_future - 1:i + n_future, 0])
            return np.array(dataX), np.array(dataY)

        X_train, y_train = split_data(train_data, n_past, n_future)
        X_test, y_test = split_data(test_data, n_past, n_future)

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        #X_train.shape, y_train.shape

        #X_test.shape, y_test.shape

        model=Sequential()
        model.add(LSTM(128,return_sequences=True,input_shape=(n_past,1)))
        model.add(LSTM(64,return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, epochs = 500, batch_size=16, validation_data=(X_test, y_test), verbose=1, callbacks=es)

        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)


        train_predict = train_predict.flatten()
        test_predict = test_predict.flatten()

        fun_df = df1.copy()

        fun_df["Train Predicted Close"] = None
        fun_df["Test Predicted Close"] = None

        

        fun_df.dtypes

        fun_df = fun_df.astype("float")




        fun_df["Train Predicted Close"].iloc[n_past:training_size]

        len(fun_df["Test Predicted Close"].iloc[training_size+n_past:])

        fun_df["Test Predicted Close"].iloc[training_size+n_past:]

        fun_df["Train Predicted Close"].iloc[n_past:training_size] = train_predict

        fun_df["Test Predicted Close"].iloc[training_size+n_past:] = test_predict
        
        final_n = test_data[-n_past:].flatten().reshape(n_future, n_past, test_data.shape[1])

        tmr_val = scaler.inverse_transform(model.predict(final_n))
        tmr_val = tmr_val.item()
        #fun_df = fun_df.asfreq("B")
        fun_df.loc[fun_df.index.max()+1*fun_df.asfreq("B").index.freq] =[None, None, tmr_val]
        
        fig = px.line(fun_df, title=stock_picked, labels={'variable':"Lines", "value":"Close"})
        fig.update_layout(title_x=0.5, hovermode="x", showlegend=True, plot_bgcolor="White", paper_bgcolor="Black",
                             font=dict(family="Courier New, monospace", size=20, color="White"))

       
        
        
        return fig





if __name__ == '__main__':
    app.run_server()
