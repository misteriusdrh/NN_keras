#SOURCE : https://github.com/Rachnog/Deep-Trading/blob/master/habrahabr.ipynb
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tools.eval_measures import rmse


df = pandas.read_csv('airline-passengers.csv')
df.Month = pandas.to_datetime(df.Month)
df = df.set_index("Month")

train, test = df[:-12], df[-12:]

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=50)
pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch = numpy.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)


df_predict = pandas.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-n_input:].index, columns=['Prediction'])

df_test = pandas.concat([df,df_predict], axis=1)

plt.isinteractive()
plt.plot(df_test.index, df_test['Passengers'])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)


#--------------------------------------
# REAL PREDICTION
#--------------------------------------

pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
print("rmse: ", pred_actual_rmse)

train = df

scaler.fit(train)
train = scaler.transform(train)

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

model.fit_generator(generator,epochs=50)

pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch = numpy.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

from pandas.tseries.offsets import DateOffset
add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pandas.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pandas.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pandas.concat([df,df_predict], axis=1)



plt.plot(df_proj.index, df_proj['Passengers'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

