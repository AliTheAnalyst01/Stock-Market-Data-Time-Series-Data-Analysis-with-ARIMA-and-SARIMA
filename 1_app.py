import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#Title
app_name = 'Stock Market Forcasting APP'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company')
# add an image online resources

st.image('https://media.istockphoto.com/id/1487894858/photo/candlestick-chart-and-data-of-financial-market.webp?b=1&s=170667a&w=0&k=20&c=iwQM0ozj7upM-_7CUEjZ2veIY3ljlB8m3PbijouIyVM=')

# take the input from the user of app about the start and end date

# sidebar
st.sidebar.header('Select the parameter from the below')

start_date = st.sidebar.date_input('Start data',date(2020,1,1))
end_date   =  st.sidebar.date_input('End data',date(2020,12,31))
# add ticker symbol list
ticker_list = ['APPL','MSFT','GOOGL','TSLA','NVD','ADBE','PYPL','INTC','INTC','CMCSA','NFLX','PEP']
ticker = st.sidebar.selectbox('Select the company',ticker_list)


# Fetch the data from user inputs using  yfinance library

data = yf.download(ticker,start=start_date,end=end_date)

#st.write(data)

# add date as a column to the datetime
data.insert(0,'Date',data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Date from ',start_date,'to',end_date)
st.write(data)

# lets plot the data
st.header('Data Visualization')
st.subheader('plot of the data')
st.write('**Note** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column')
fig = px.line(data,x='Date',y=data.columns,title='Closing Price of Stock',width=800,height=600)
st.plotly_chart(fig)

# add a select box to select colum from the data 
column = st.selectbox('Select the Column to be used for forecasting',data.columns[1:])

# subsetting the data
data = data [['Date',column]]
st.write('Selected Data')
st.write(data)

# ADF test check stationarity
st.header('Is data stationarity?')
st.write('**Note** If p-value is less than 0.05 then data is stationarity')
st.write(adfuller(data[column])[1] < 0.05)

# lets decompose the data
st.header('Decomposition of the data column')
decomposition = seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

# lets run the model 
# user input for three parameter of the model and seasonal order

p =  st.slider('Select the value of p',0,5,2)
d = st.slider('Select the value of d',0,5,1)
q = st.slider('Select the value of q',0,5,2)
seasonal_order = st.number_input('Select the value of seasonal p',0,24,12)


model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# print the model summary
st.header('Model Summary')
st.write(model.summary())
st.write('---')


# predict the future value(Forecasting)
st.write("<p style='color:green; font-size:50px; font0-weight:bold;'>Forecaseting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input('Select the number of the days to forecaset',value=10)
# predict the future values
predictions = model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions = predictions.predicted_mean
#st.write(predictions)


# add index to result dataframe as dates
predictions.index = pd.date_range(start = end_date,periods=len(predictions),freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index)
predictions.reset_index(drop=True,inplace=True)
st.write("## Predictions",predictions)
st.write('## Actual Date',data)
st.write('---')

# lets plot the data
fig = go.Figure()
# add the actual data to the plot 
fig.add_trace(go.Scatter(x=data['Date'],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
# add the predicted data to the plot
fig.add_trace(go.Scatter(x=predictions['Date'],y=predictions['predicted_mean'],mode='lines',name='Predictied',line=dict(color='red')))
# set the title and axis labels
fig.update_layout(title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Price',width=800,height=400)
# display the plot
st.plotly_chart(fig)

st.write('---')

st.write('### About the AUthur')

st.write("<p style='color:blue; font-weight:bold ; font-size:50px;'>Syed Ali Faizan</p>",unsafe_allow_html=True)

st.write('## Connect with me on social media')
# add links to my social media
# url of the images

linkedin_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVU5JBJYMzdr32kvZCr6eoRT5xy3E91kbUaTvAEr0&s"
github_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVdtrueQqt-wsYtp-UI1DutqtUbtrEeeKDtXAJE0U&s"
kaggle_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQnB1IbBU743XF3Aipf0kZhcTixQV3IARu27rZTPaEfg&s"

# redirect urls
linkedin_redirect_url = "www.linkedin.com/in/syed-ali-faizan-5131bb194"
github_redirect_url = "https://github.com/AliTheAnalyst01"
kaggle_redirect_url = "https://www.kaggle.com/faizanzaidy"

# add the links to the images
st.markdown(f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
            f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
            f'<a href="{kaggle_redirect_url}"><img src="{kaggle_url}" width="60" height="60"></a>',unsafe_allow_html=True)