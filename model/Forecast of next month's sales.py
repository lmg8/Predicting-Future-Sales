#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

train = pd.read_csv('../input/sales_train_v2.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')


# In[2]:


train.head()


# In[3]:


print ('number of shops: ', train['shop_id'].max())
print ('number of items: ', train['item_id'].max())
print ('size of train: ', train.shape)
print('max date:', train.date.max())
print('min date:', train.date.min())


# In[4]:

#graph
grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1


# In[5]:


#get data starting july 2013 - unlikely it'll will affect sales of Nov 2015
train = train.loc[train['date_block_num'] >= 18]
print(train.head())


# In[6]:


data = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0)
data.reset_index(inplace = True)


# In[7]:


data.head()


# In[8]:


#merge pivot table with the test 
#want to keep the data of items we have predict
data = pd.merge(test,data,on = ['item_id','shop_id'],how = 'left')


# ### Missing Data

# Fill NaN values in data with 0

# In[9]:


data.fillna(0,inplace = True)


# In[10]:


data.head()


# In[11]:


# X we will keep all columns except the last one 
X_train = np.expand_dims(data.values[:, :-1], axis=2)

# the last column is our label
y_train = data.values[:,-1:]

# for test we keep all the columns except the first one
X_test = np.expand_dims(data.values[:,1:],axis = 2)

print(X_train.shape,y_train.shape,X_test.shape)


#LSTM Model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout


# In[13]:


l_model = Sequential()
l_model.add(LSTM(units = 64,input_shape = (18,1)))
l_model.add(Dropout(0.3))
l_model.add(Dense(1))

l_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
l_model.summary()


# In[14]:


history = l_model.fit(X_train,y_train, epochs=10, batch_size=4096)


# In[15]:



plt.plot(history.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(history.history['mean_squared_error']), label= 'rmse')
plt.legend(loc=1)


# In[16]:


test.head()


# In[17]:


#Prediction
sub_pfs = l_model.predict(X_test)
prediction = pd.DataFrame({'ID':test['ID'],'shop_id':test['shop_id'],'item_id':test['item_id'],'item_cnt_month':sub_pfs.ravel()})
prediction.to_csv('forecast.csv',index=False)

