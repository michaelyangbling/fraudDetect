import tensorflow as tf
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
data_path="/Users/yzh/Desktop/fraudDetect/"

train=pd.read_csv( data_path+'train.csv.zip', dtype={'ip':'int32','app':'int16','device':'int16',
                  'os':'int16','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution

test=pd.read_csv( data_path+'test.csv.zip', dtype={'ip':'int32','app':'int16','device':'int16',
                  'os':'int16','channel':'int16'}
                    ,parse_dates=['click_time'])
print("finished loading")
test['click_hour']=test['click_time'].dt.hour.astype(np.int8) #turn into hour
train['click_hour']=train['click_time'].dt.hour.astype(np.int8)

from sklearn.preprocessing import StandardScaler
timeScaler=StandardScaler()  # feature scaling
timeScaler.fit(train[["click_hour"]])
test['scaledHour']=timeScaler.transform(test[["click_hour"]])
train['scaledHour']=timeScaler.transform(train[["click_hour"]])

train.drop(['click_time'],axis=1,inplace=True)
test.drop(['click_time'],axis=1,inplace=True)
trn=train[0:160000000]
tst=train[160000000:(train.shape[0])]
del train
del test
gc.collect()
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

myFeatureColumns.append(tf.feature_column.bucketized_column(
    source_column = tf.feature_column.numeric_column("click_hour"), # bucketize time
    boundaries = [2.5,5.5, 8.5,11.5,14.5,17.5,20.5])
)
myFeatureColumns.append(tf.feature_column.numeric_column(key='scaledHour'))
def getWeight(train): # set weight for imbalanced class to do imbalanced trainng
    num1=train.is_attributed.sum()
    if num1==0 or num1==train.shape[0]: #only 0 in training set label
        print("only one class in training set label ")
    else:
        weight1=(train.shape[0]-num1)/num1
        print(weight1)
        train['weight']=train.is_attributed.apply(lambda x: weight1 if x==1 else 1)
getWeight(trn)
print("weight allocated")
#trn.loc[trn.is_attributed==True,'weight'] #check correctness
#myFeatureColumns.append(tf.feature_column.numeric_column(key='weight'))
import logging
logging.getLogger().setLevel(logging.INFO)

#then conduct some embedding to reduce features~
featuCol2=[]
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
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2,
    weight_column=tf.feature_column.numeric_column(key='weight'))
gc.collect()
classifier2.train(
    input_fn=tf.estimator.inputs.pandas_input_fn(trn,y=trn.is_attributed,
                            batch_size=128,shuffle=True,num_threads=4),steps=30000) # batch size=1000,
gc.collect()
check=tst.tail(60000)
print("label_mean"+str(check.is_attributed.mean()))
predictions2=classifier2.predict(tf.estimator.inputs.pandas_input_fn(check,shuffle=False))
probs2=[]
for i in predictions2:
  probs2.append(i['probabilities'][1])
from sklearn.metrics import roc_curve, auc
a2,b2,c2=roc_curve(check['is_attributed'],probs2)
print((auc(a2,b2)))
# some  data analysis
# print(pd.isnull(train).sum())  #no na in test data
# print(pd.isnull(test).sum())   #no na in train data

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

# we are lucky to have a clean dataset!

# np.unique(train.app)
# array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
#         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
#         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
#         39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
#         52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
#         65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
#         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
#         91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
#        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
#        117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
#        130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#        143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
#        169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
#        182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
#        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
#        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#        221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
#        234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
#        247, 248, 249, 250, 251, 252, 253, 254, 255], dtype=uint8)
#
# np.unique(train.ip)
# array([     1,      5,      6, ..., 364776, 364777, 364778], dtype=int32)

#np.unique(train.ip).shape
# (277396,)


# train.describe()
#                  ip           app        device            os       channel  \
# count  1.849039e+08  1.849039e+08  1.849039e+08  1.849039e+08  1.849039e+08
# mean   9.087604e+04  1.201131e+01  2.172325e+01  2.267702e+01  2.685789e+02
# std    6.952789e+04  1.480521e+01  2.593326e+02  5.525282e+01  1.295882e+02
# min    1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
# 25%    4.024500e+04  3.000000e+00  1.000000e+00  1.300000e+01  1.400000e+02
# 50%    7.962200e+04  1.200000e+01  1.000000e+00  1.800000e+01  2.580000e+02
# 75%    1.182470e+05  1.500000e+01  1.000000e+00  1.900000e+01  3.790000e+02
# max    3.647780e+05  7.680000e+02  4.227000e+03  9.560000e+02  5.000000e+02
#          click_hour    scaledHour
# count  1.849039e+08  1.849039e+08
# mean   9.298776e+00  8.819965e-15
# std    6.171641e+00  1.000000e+00
# min    0.000000e+00 -1.506694e+00
# 25%    4.000000e+00 -8.585685e-01
# 50%    9.000000e+00 -4.841115e-02
# 75%    1.400000e+01  7.617462e-01
# max    2.300000e+01  2.220029e+00