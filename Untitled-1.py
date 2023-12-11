import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
path = './tdd.csv'
data = pd.read_csv(path)

df = pd.DataFrame(data)
df.drop([ 'Trip ID', 'Destination','Start date','Traveler nationality', 'End date', 'Traveler name', 'Transportation type', 'Transportation cost'], axis=1,inplace=True) #, 'Transportation cost'
#df.rename(['Traveler_gender','Traveler_age','Accommodation_type', 'Accommodation_cost'], inplace=True)
df.dropna(inplace=True)
#df.rename(columns={'Duration (days)': 'Duration'}, inplace=True)
df.columns=['Duration', 'Traveler_age', 'Traveler_gender', 'Accommodation_type', 'Accommodation_cost']

df['Traveler_ages'] = df['Traveler_age'].agg(lambda x: x - x%10).map(int)
df['Accommodation_cost'] = df['Accommodation_cost'].agg(lambda x: re.sub('[^0-9]', '', x)).map(int)

sns.set_style('whitegrid')

#성별과 나이에 따른 평균 여행기간(일)
grouped1 = df.groupby(['Traveler_gender', 'Traveler_ages'])
dumean = grouped1['Duration'].mean()

#성별 간 숙박업소에 대한 파이차트와 비용에 대한 막대그래프
grouped2 = df.groupby(['Traveler_gender', 'Accommodation_type'])
#minmax scale
acc_type = grouped2.mean().Accommodation_cost.to_frame()

def minmax(x):
    return (x - x.min())/(x.max() - x.min())

f_scale = acc_type.loc["Female"].transform(minmax)
m_scale = acc_type.loc['Male'].transform(minmax)

#grouped3 = df.groupby('Traveler_gender').get_group('Male')
#g3_mean = grouped3.groupby(('Accommodation_type')).mean()


############### 그래프 2가지 이상
fig =  plt.figure(figsize=(13,8))
#-> 나이에 따른 여행기간의 바이올린그래프 
ax1 = fig.add_subplot(1, 2, 1)
sns.violinplot(x = 'Traveler_gender', y = 'Traveler_ages', hue='Traveler_gender', data=dumean.to_frame(), ax=ax1)

ax2 = fig.add_subplot(1, 2, 2)
sns.distplot(dumean.to_frame(), ax=ax2)

#파이차트
fig_pie = plt.figure(figsize=(15, 10))

ax3 = fig_pie.add_subplot(1, 2, 1)
f_scale.plot(kind='pie', figsize=(13,7), subplots=True, ax=ax3, title='Accommdation', legend=False)
ax4 = fig_pie.add_subplot(1, 2, 2)
plt.pie(data=m_scale, autopct='%1.1f%%',labels=acc_type.loc['Male'].index, x='Accommodation_cost')
#sns.lmplot(x='Accommodation type', y = 'Accommodation cost', data=m_scale, hue=g3_mean.index)
plt.axis('equal')
plt.legend()

#회귀분석

ndf = df[['Accommodation_cost', 'Traveler_age', 'Duration' ]]
 
#g3_mean.plot(kind='scatter', x='Traveler_ages', y='Accommodation_cost', s = 10, figsize=(10, 5))
ndf.plot(kind='scatter', x='Accommodation_cost', y='Traveler_age', s=10, figsize=(10, 5))

X = ndf[['Traveler_age']]
Y = ndf['Duration'] 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
#print(len(X_train), len(Y_train)) #95 95

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)



y_hat = lr.predict(X)
#mse
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(X, y_hat)
print('mse')
print(mse)

plt.figure(figsize=(10, 5))

ax1 = sns.distplot(Y, hist=False,label='y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax=ax1)
plt.show()

plt.close()