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

def input_func(train):
    features={'ip':np.array(train.ip),'app':np.array(train.app),'device':np.array(train.device),
      'os':np.array(train.os),'channel':np.array(train.time),'click_hour':np.array(train.click_hour),
              'scaledHour':np.array(train.scaledHour)}
    labels=np.array(train.is_attributed)
    return features, labels

trainSmall = train.sample(5 * (10 ** 5))

trn=trainSmall.head(4*(10 ** 5))
myFeatureColumns=[]
myFeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list( key='ip',vocabulary_list=trn.ip.unique() ))
myFeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list( key='app',vocabulary_list=trn.app.unique() ))
myFeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list( key='device',vocabulary_list=trn.device.unique() ))
myFeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list( key='os',vocabulary_list=trn.os.unique() ))
myFeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list( key='channel',vocabulary_list=trn.channel.unique() ))

myFeatureColumns.append(tf.feature_column.bucketized_column(
    source_column = tf.feature_column.numeric_column("click_hour"), # bucketize time
    boundaries = [2.5,5.5, 8.5,11.5,14.5,17.5,20.5])
)
myFeatureColumns.append(tf.feature_column.numeric_column(key='scaledHour'))
tst=trainSmall.tail(1*(10 ** 5))

def getWeight(train): # set weight for imbalanced class
    num1=train.is_attributed.sum()
    if num1==0: #only 0 in training set label
        print("only 0 in training set label ")
    else:
        weight1=(train.shape[0]-num1)/num1
        print(weight1)
        train['weight']=train.is_attributed.apply(lambda x: weight1 if x==1 else 1)

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

# np.unique(train.device)
# array([-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118,
#        -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107,
#        -106, -105, -104, -103, -102, -101, -100,  -99,  -98,  -97,  -96,
#         -95,  -94,  -93,  -92,  -91,  -90,  -89,  -88,  -87,  -86,  -85,
#         -84,  -83,  -82,  -81,  -80,  -79,  -78,  -77,  -76,  -75,  -74,
#         -73,  -72,  -71,  -70,  -69,  -68,  -67,  -66,  -65,  -64,  -63,
#         -62,  -61,  -60,  -59,  -58,  -57,  -56,  -55,  -54,  -53,  -52,
#         -51,  -50,  -49,  -48,  -47,  -46,  -45,  -44,  -43,  -42,  -41,
#         -40,  -39,  -38,  -37,  -36,  -35,  -34,  -33,  -32,  -31,  -30,
#         -29,  -28,  -27,  -26,  -25,  -24,  -23,  -22,  -21,  -20,  -19,
#         -18,  -17,  -16,  -15,  -14,  -13,  -12,  -11,  -10,   -9,   -8,
#          -7,   -6,   -5,   -4,   -3,   -2,   -1,    0,    1,    2,    3,
#           4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,
#          15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,
#          26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,
#          37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,
#          48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
#          59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,
#          70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,
#          81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,
#          92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,
#         103,  104,  105,  106,  107,  108,  109,  110,  111,  112,  113,
#         114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,
#         125,  126,  127], dtype=int8)
#
# np.unique(train.os)
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
# np.unique(train.channel)
# array([  0,   3,   4,   5,  13,  14,  15,  17,  18,  19,  21,  22,  24,
#         29,  30, 101, 105, 107, 108, 110, 111, 113, 114, 115, 116, 118,
#        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 134, 135,
#        137, 138, 140, 142, 145, 146, 149, 150, 153, 160, 162, 165, 169,
#        171, 172, 173, 174, 178, 181, 182, 203, 205, 208, 210, 211, 212,
#        213, 215, 216, 217, 219, 221, 222, 223, 224, 225, 227, 232, 233,
#        234, 236, 237, 238, 242, 243, 244, 245, 248, 251, 253, 256, 258,
#        259, 261, 262, 265, 266, 268, 272, 274, 277, 278, 280, 281, 282,
#        311, 315, 317, 319, 320, 322, 325, 326, 328, 330, 332, 333, 334,
#        340, 341, 343, 347, 349, 352, 353, 354, 356, 360, 361, 364, 371,
#        373, 376, 377, 379, 386, 391, 394, 400, 401, 402, 404, 406, 407,
#        408, 409, 410, 411, 412, 414, 416, 417, 419, 420, 421, 422, 424,
#        430, 434, 435, 439, 442, 445, 446, 448, 449, 450, 451, 452, 453,
#        455, 456, 457, 458, 459, 460, 463, 465, 466, 467, 469, 470, 471,
#        473, 474, 475, 476, 477, 478, 479, 480, 481, 483, 484, 486, 487,
#        488, 489, 490, 496, 497, 498, 500], dtype=int16)
