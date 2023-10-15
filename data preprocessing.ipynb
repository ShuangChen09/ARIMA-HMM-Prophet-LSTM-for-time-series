import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import scipy.stats as stats
import sys
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from hmmlearn import hmm

#Define the model evaluation metric function
def MSE(a, b):
    c=a-b
    d=c*c
    sum =np.mean(d)
    return sum
def MADE(a,b):
    c=a-b
    d=abs(c)
    sum=np.mean(d)
    return sum

#Read data
data=pd.read_excel(r"C:\Users\LX\Desktop\时间序列\案例分析报告\usunemployment.xlsx")
data.head()

#Data preprocessing, format adjustment
df=pd.melt(data,id_vars=['Year'],var_name='Month',value_name='Value')
df['Date']=pd.to_datetime(df['Year'].astype(str)+'-'+df['Month']).dt.strftime('%Y-%m')
df=df.drop(['Year', 'Month'],axis=1)
df=df.sort_values('Date')
df=df.iloc[:, ::-1]
df=df.dropna(axis=0)
print(df)
df.to_excel('C:\\Users\\LX\\Desktop\\时间序列\\案例分析报告\\dataset.xlsx)

newdf=pd.read_excel(r"C:\Users\LX\Desktop\时间序列\案例分析报告\dataset.xlsx")
newdf=newdf.iloc[:,1:3]



