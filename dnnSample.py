import tensorflow as tf
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
data_path="/Users/yzh/Desktop/fraudDetect/"
train=pd.read_csv( data_path+'train_sample.csv', dtype={'ip':'int32','app':'int16','device':'int16',
                  'os':'int16','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution
#turn into hour
train['click_hour']=train['click_time'].dt.hour.astype(np.int8)
from sklearn.preprocessing import StandardScaler
timeScaler=StandardScaler()  # feature scaling
timeScaler.fit(train[["click_hour"]])
train['scaledHour']=timeScaler.transform(train[["click_hour"]])
def input_func(train,batch_size):
    df=train.sample(batch_size)
    features={'ip':np.array(train.ip),'app':np.array(train.app),'device':np.array(train.device),
      'os':np.array(train.os),'channel':np.array(train.channel),'click_hour':np.array(train.click_hour),
              'scaledHour':np.array(train.scaledHour),'weight':np.array(train.weight)}
    labels=np.array(train.is_attributed)
    return features, labels
trn=train.head(80000)
myFeatureColumns=[]
myFeatureColumns.append( tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='ip',vocabulary_list=trn.ip.unique() )))
myFeatureColumns.append( tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='app',vocabulary_list=trn.app.unique() )))
myFeatureColumns.append( tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='device',vocabulary_list=trn.device.unique() )))
myFeatureColumns.append(tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='os',vocabulary_list=trn.os.unique() )))
myFeatureColumns.append(tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list( key='channel',vocabulary_list=trn.channel.unique() )))
myFeatureColumns.append(tf.feature_column.numeric_column(key='scaledHour'))
myFeatureColumns.append(tf.feature_column.bucketized_column(
    source_column = tf.feature_column.numeric_column("click_hour"), # bucketize time
    boundaries = [2.5,5.5, 8.5,11.5,14.5,17.5,20.5])
)
tst=train.tail(20000)
def getWeight(train): # set weight for imbalanced class to do imbalanced trainng
    num1=train.is_attributed.sum()
    if num1==0: #only 0 in training set label
        print("only 0 in training set label ")
    else:
        weight1=(train.shape[0]-num1)/num1
        print(weight1)
        train['weight']=train.is_attributed.apply(lambda x: weight1 if x==1 else 1)
getWeight(trn)
#trn.loc[trn.is_attributed==True,'weight'] #check correctness
myFeatureColumns.append(tf.feature_column.numeric_column(key='weight'))
classifier = tf.estimator.DNNClassifier(
    feature_columns=myFeatureColumns,
    model_dir="model",
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2,
    weight_column='weight')

classifier.train(
    input_fn=lambda:input_func(trn),steps=200)