import dash
import pandas_datareader.data as web
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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
from pmdarima import auto_arima
from datetime import datetime
import chart_studio.plotly as py
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf

import plotly.graph_objects as go
app = dash.Dash()
server = app.server

app.layout = html.Div([
                    
                    html.H1("Stock ticker Dashboard", style={"text-align":"center"}),
    
    
                    html.Div([html.H3("Select Desired Stocks",
                                      style={"paddingRight":"30px"}),
    
                                dcc.Input(id="stockpicker", value="TSLA",
                                         style={"fontSize":24,"width":75})
                             ],
                            style={"display":"inline-block", "verticalAlign":"top"}
                            ),
    
    
                    html.Div([html.H3("Select Date Range"),
                             dcc.DatePickerRange(id="datepicker", min_date_allowed=datetime(2015,1,1),
                                                                    max_date_allowed=datetime(2025,1,1),
                                                                    start_date=datetime(2018,1,1),
                                                                    end_date = datetime.today(),
                                                                    display_format="D-M-Y") 
                             ],
                            style={"display":"inline-block"}
                            ),
                    
    
                    html.Div([html.Button(id="submit_button", n_clicks=0, children="Submit", style={"fontSize": 24})
                                          ],
                                         
                            style={"display":"inline-block", "border":"2px solid black"}),
    
                    
                    dcc.Graph(id="graph")
    
    
                    
                            
                            
])




@app.callback(Output("graph", "figure"),
              [Input("submit_button", "n_clicks")],
              
             [State("stockpicker", "value"),
              State("datepicker","start_date"), 
              State("datepicker","end_date")])

#datepickerrange internally datetime but when you grab it becomes string representation so much change back



def update_dashboard(n_clicks, selected_ticker, start_date, end_date):
    
    df = web.DataReader(selected_ticker, "yahoo", start=start_date, end=end_date)
    #df = web.DataReader(selected_ticker, "yahoo", start=datetime(2017,1,1), end=datetime(2017,12,31))
    fig = px.line(df, x=df.index, y="Close", title=selected_ticker)
    fig.update_layout(title_x=0.5, hovermode="x")
    return fig 





if __name__ == '__main__':
    app.run_server()
