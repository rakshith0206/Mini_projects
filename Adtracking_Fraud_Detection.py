
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

path = '../input/'

dtypes = {'ip'            : 'uint32',
          'app'           : 'uint16',
          'device'        : 'uint16',
          'os'            : 'uint16',
          'channel'       : 'uint16',
          'is_attributed' : 'uint8',
          'click_id'      : 'uint32' }
          
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, skiprows = range(1, 131886954), usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
train_df.head()
train_df.shape

pd.value_counts(train_df['is_attributed'].values,sort=False)
train_df['hour']=pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

print('grouping by ip-day-hour combination...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str,columns={'channel':'qty'})
train_df = train_df.merge(gp,on=['ip','day','hour'],how='left')
train_df.head()

print('group by ip-app combination....')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
train_df.head()

print('group by ip-app-os combination....')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
train_df.head()

print("vars and data type....")
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

train_df.dtypes
train_df['is_attributed'] = train_df['is_attributed'].astype('object')
train_df.dtypes

from sklearn.preprocessing import LabelEncoder
train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
y_train = train_df['is_attributed']
train_df.drop(['click_time','ip','is_attributed'],1,inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_df,y_train,test_size = 0.20,random_state=42)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

del train_df

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

max_app = np.max([X_train['app'].max(), X_test['app'].max()])+1
max_ch = np.max([X_train['channel'].max(), X_test['channel'].max()])+1
max_dev = np.max([X_train['device'].max(), X_test['device'].max()])+1
max_os = np.max([X_train['os'].max(), X_test['os'].max()])+1
max_h = np.max([X_train['hour'].max(), X_test['hour'].max()])+1
max_d = np.max([X_train['day'].max(), X_test['day'].max()])+1
max_wd = np.max([X_train['wday'].max(), X_test['wday'].max()])+1
max_qty = np.max([X_train['qty'].max(), X_test['qty'].max()])+1
max_c1 = np.max([X_train['ip_app_count'].max(), X_test['ip_app_count'].max()])+1
max_c2 = np.max([X_train['ip_app_os_count'].max(), X_test['ip_app_os_count'].max()])+1

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X

X_train = get_keras_data(X_train)
X_train

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h)
in_d = Input(shape=[1], name = 'd')
emb_d = Embedding(max_d, emb_n)(in_d)
in_wd = Input(shape=[1], name = 'wd')
emb_wd = Embedding(max_wd, emb_n)(in_wd)
in_qty = Input(shape=[1], name = 'qty')
emb_qty = Embedding(max_qty, emb_n)(in_qty)
in_c1 = Input(shape=[1], name = 'c1')
emb_c1 = Embedding(max_c1, emb_n)(in_c1)
in_c2 = Input(shape=[1], name = 'c2')
emb_c2 = Embedding(max_c2, emb_n)(in_c2)

fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
s_dout = SpatialDropout1D(0.2)(fe)
x = Flatten()(s_dout)
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd,in_qty,in_c1,in_c2], outputs=outp)
batch_size = 20000
epochs = 2

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=2, shuffle=True, verbose=2)

X_test = get_keras_data(X_test)
X_test

Y_pred = model.predict(X_test,batch_size=batch_size, verbose=2)
Y_pred.dtype
Y_pred = Y_pred.astype('object')















































