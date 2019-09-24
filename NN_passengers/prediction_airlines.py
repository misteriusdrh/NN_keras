#SOURCE : https://github.com/Rachnog/Deep-Trading/blob/master/habrahabr.ipynb
import matplotlib.pyplot as plt
import numpy
import pandas
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=9, batch_size=2, verbose=2)


# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

listPredict = numpy.append (trainPredict, testPredict)

plt.plot(dataset)

plt.plot(listPredict)
plt.plot(trainPredict)
plt.plot(testPredict)

plt.isinteractive()
plt.show()