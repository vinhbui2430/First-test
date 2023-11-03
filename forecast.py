import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


weather_df = pd.read_csv('Hanoi_weather.csv', parse_dates=['date'], index_col='date')
weather_df.head(5) # Show the first 5 rows

weather_df.columns

weather_df.shape 

weather_df.describe()

weather_df.isnull().any()

weather_df_num=weather_df.loc[:,['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure']]
weather_df_num.head()

weather_df_num.shape

weather_df_num.columns
weather_df_num.plot(subplots=True, figsize=(25,20))

weather_df_num = weather_df_num.sort_index().loc['2019' : '2020'] 

weather_df_num.index = pd.to_datetime(weather_df_num.index, format='mixed', dayfirst=True)
weather_df_num['2019':'2020'].resample('D').fillna(method= 'pad').plot(subplots=True, figsize=(25,20))

weather_df_num.hist(bins=10,figsize=(15,15))

weth=weather_df_num['2019':'2020']
weth.head()

weather_y=weather_df_num.pop('max')
weather_x=weather_df_num

train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)

train_X.shape
train_y.shape

train_y.head()

plt.scatter(weth['min'], weth['max'])
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.show()

plt.scatter(weth['pressure'], weth['min'])
plt.xlabel("Minimum Temperature")
plt.ylabel("Temperature")
plt.show()

plt.scatter(weth['min'], weth['max'])
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.show()

model=LinearRegression()
model.fit(train_X,train_y)

prediction = model.predict(test_X)

#calculating error
np.mean(np.absolute(prediction-test_y))

print('Variance score: %.2f' % model.score(test_X, test_y))

for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction,'diff':(test_y-prediction)})

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(train_X,train_y)

prediction2=regressor.predict(test_X)
np.mean(np.absolute(prediction2-test_y))

print('Variance score: %.2f' % regressor.score(test_X, test_y))

for i in range(len(prediction2)):
  prediction2[i]=round(prediction2[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction2,'diff':(test_y-prediction2)})

from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=90,random_state=0,n_estimators=100)
regr.fit(train_X,train_y)
prediction3=regr.predict(test_X)
np.mean(np.absolute(prediction3-test_y))
print('Variance score: %.2f' % regr.score(test_X, test_y))
for i in range(len(prediction3)):
  prediction3[i]=round(prediction3[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction3,'diff':(test_y-prediction3)})

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction ) )
print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction2 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction2 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction2 ) )

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction3 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction3 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,prediction3 ) )