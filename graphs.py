from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

df1 = pd.read_csv(r"C:\Users\lqm\Desktop\data\sales_train_v2.csv")

df2 = pd.read_csv(r"C:\Users\lqm\Desktop\data\items1.csv")

item_name=df1['item_id']

price=df1['item_price']

category0=df2['item_category_id']

category=list(range(84))

category_money=[0]*84

fig=plt.figure()

fig1=plt.figure()

fig2=plt.figure()

fig3=plt.figure()

ax=fig.add_axes([0,0,1,1])

bx=fig1.add_axes([0,0,1,1])

cx=fig2.add_axes([0,0,1,1])

dx=fig3.add_axes([0,0,1,1])

cnt=df1['item_cnt_day']

date=df1['date_block_num']

shop1=list(range(60))

number_sold=[0]*60

shop_id=df1['shop_id']

shop_id=df1['shop_id']

sales=df1['item_cnt_day']

id1=[]

cnt1=[]

date1=list(range(34))

date_number_sold=[0]*34

date_money=[0]*34

shop_money=[0]*60

for i in range(0,10):

   number_sold[shop_id[i]]=number_sold[shop_id[i]]+1

   date_number_sold[date[i]]=date_number_sold[date[i]]+1

   date_money[date[i]]=date_money[date[i]]+price[i]*cnt[i]

   shop_money[shop_id[i]]=shop_money[shop_id[i]]+price[i]*cnt[i]

   category_money[category0[item_name[i]]]=category_money[category0[item_name[i]]]+price[i]*cnt[i]

dx.bar(category,category_money,1)

dx.title.set_text('category-sales')

dx.set_xlabel('category')

dx.set_ylabel('sales')

cx.plot(shop1,shop_money,color='red')

cx.title.set_text('store-sales')

cx.set_xlabel('store#')

cx.set_ylabel('sales')

bx.plot(shop1,number_sold,color='y')

bx.title.set_text('#of items sold from 2013 to 2015')

bx.set_xlabel('store')

bx.set_ylabel('sales')

ax.plot(date1,date_money,color='m')

ax.title.set_text('sales performance')

ax.set_xlabel('month')

ax.set_ylabel('sales')

Message Input





Message #general
