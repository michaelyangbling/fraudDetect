import tensorflow as tf
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
#chinese time is UTC +8, UTC time need to plus 8 hours in dataframe
data_path="/Users/yzh/Desktop/fraudDetect/"
train=pd.read_csv( data_path+'train_sample.csv', dtype={'ip':'int32','app':'int16','device':'int16',
                  'os':'int16','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution
#turn into hour
train['click_time']=train['click_time']+pd.Timedelta(hours=8)
train['click_hour']=train['click_time'].dt.hour.astype(np.int8)
from sklearn.preprocessing import StandardScaler
timeScaler=StandardScaler()  # feature scaling
timeScaler.fit(train[["click_hour"]])
train['scaledHour']=timeScaler.transform(train[["click_hour"]])
start = pd.to_datetime('11/06/2017 00:00')
train['time_diff']=(train.click_time-start).dt.total_seconds()
train['difDays']=(train.click_time.dt.normalize()-start).dt.total_seconds()/(3600*24)

trn=train[train['click_time'] < datetime.date(2017,11,9)]
tst=train[train['click_time'] >= datetime.date(2017,11,9)]

difScaler=StandardScaler();difScaler.fit(train[['time_diff']])
trn['sDiff']=difScaler.transform(trn[['time_diff']]);tst['sDiff']=difScaler.transform(tst[['time_diff']])

dayScaler=StandardScaler();dayScaler.fit(train[['difDays']])
trn['dayDiff']=dayScaler.transform(trn[['difDays']]);tst['dayDiff']=dayScaler.transform(tst[['difDays']])

def getWeight(train): # set weight for imbalanced class to do imbalanced trainng
    num1=train.is_attributed.sum()
    if num1==0: #only 0 in training set label
        print("only 0 in training set label ")
    else:
        weight1=(train.shape[0]-num1)/num1
        print(weight1)
        train['weight']=train.is_attributed.apply(lambda x: weight1 if x==1 else 1)
getWeight(trn)
import logging
logging.getLogger().setLevel(logging.INFO)
#trn=train.head(80000)





#then conduct some embedding to reduce features~
featuCol2=[]
#featuCol2.append(tf.feature_column.numeric_column(key='sDiff'))
#featuCol2.append(tf.feature_column.numeric_column(key='dayDiff'))

featuCol2.append( tf.feature_column.embedding_column(categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
    key='ip',vocabulary_list=trn.ip.unique() ),dimension=40))
featuCol2.append( tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='app',vocabulary_list=trn.app.unique() )))
featuCol2.append( tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='device',vocabulary_list=trn.device.unique() )))
featuCol2.append(tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='os',vocabulary_list=trn.os.unique() )))
featuCol2.append(tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='channel',vocabulary_list=trn.channel.unique() )))

featuCol2.append(tf.feature_column.bucketized_column(
    source_column = tf.feature_column.numeric_column("click_hour"), # bucketize time
    boundaries = [2.5,5.5, 8.5,11.5,14.5,17.5,20.5])
)
featuCol2.append(tf.feature_column.numeric_column(key='scaledHour'))
classifier2 = tf.estimator.DNNClassifier(
    feature_columns=featuCol2,
    model_dir=None,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10,10],
    # The model must choose between 3 classes.
    n_classes=2,
    activation_fn=tf.nn.elu,
    weight_column=tf.feature_column.numeric_column(key='weight'),
    dropout=0.5,
    #dropout=0.5 #regularize
    )

from sklearn.metrics import roc_curve, auc
inputF=tf.estimator.inputs.pandas_input_fn(trn.drop('click_time',axis=1),y=trn.is_attributed,
                            batch_size=128,shuffle=True,num_threads=4)
tstN=tst.shape[0]
tstInput=tf.estimator.inputs.pandas_input_fn(tst.drop('click_time', axis=1), shuffle=False,batch_size=128)
validAuc=0
n=0
for hid in ([35],[100],[10,10],[80,40],[30,20,10],[60,40,20],[40,20,10],[50,25,10,5],[40,30,20,10,5]):
  for activ in (tf.nn.elu, tf.nn.relu):
      for
  while True: #early stopping using validation set
    classifier2.train(input_fn=inputF,steps=200)
    n = n + 1
    if n<=4:
        continue
    predictions2 = classifier2.predict(input_fn=tstInput)
    probs2 = []
    for i in predictions2:
        probs2.append(i['probabilities'][1])
    a2, b2, c2 = roc_curve(tst['is_attributed'], probs2)
    aucx=auc(a2, b2)
    if aucx<=validAuc+0.00002: #0.000001
        print("auc: "+str(aucx))
        break
    validAuc=aucx

# classifier2.train(
#     input_fn=tf.estimator.inputs.pandas_input_fn(trn.drop('click_time',axis=1),y=trn.is_attributed,
#                             batch_size=128,shuffle=False,num_threads=4),steps=600) # batch size=1000,
#
# predictions2=classifier2.predict(tf.estimator.inputs.pandas_input_fn(tst.drop('click_time',axis=1),shuffle=False))
# #predict: call pandas_input_fn until end?
# print(tst.is_attributed.mean())
# probs2=[]
# for i in predictions2:
#   probs2.append(i['probabilities'][1])
# from sklearn.metrics import roc_curve, auc
# a2,b2,c2=roc_curve(tst['is_attributed'],probs2)
# print("auc: "+str(auc(a2,b2)))#100 steps:0.91532859802, but speed up a lot    embedding_best: 0.994696502302   no_embedding_best:0.9932

#when using 300 steps: 0.875725720975    0.834676675156