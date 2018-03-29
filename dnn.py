import tensorflow as tf
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
data_path="/Users/yzh/Desktop/fraudDetect/"

train=pd.read_csv( data_path+'train.csv.zip', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution

test=pd.read_csv( data_path+'test.csv.zip', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16'}
                    ,parse_dates=['click_time'])
test['click_hour']=test['click_time'].dt.hour.astype(np.int8) #turn into hour
train['click_hout']=train['click_time'].dt.hour.astype(np.int8)
#print(pd.isnull(train).sum())  #no na in test data
#print(pd.isnull(test).sum())   #no na in train data

# print(pd.isnull(train).sum())
# ip               0
# app              0
# device           0
# os               0
# channel          0
# click_time       0
# is_attributed    0
# click_hour       0
# dtype: int64

# print(pd.isnull(test).sum())
# click_id      0
# ip            0
# app           0
# device        0
# os            0
# channel       0
# click_time    0
# click_hour    0
# scaledHour    0
# dtype: int64


#lucky to have a clean dataset!
from sklearn.preprocessing import StandardScaler
timeScaler=StandardScaler()  # feature scaling
timeScaler.fit(train[["click_hour"]])
test['scaledHour']=timeScaler.transform(test[["click_hour"]])
train['scaledHour']=timeScaler.transform(train[["click_hour"]])


