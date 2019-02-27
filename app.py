# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv("rotamers_han_train.csv")
dataset = dataset.dropna()
# split into input (X) and output (Y) variables
X = dataset.iloc[:,[3,4,5,6]]
X2 = dataset.iloc[:,8]

    

#X2 = X2.reshape(1,-1)
#X2 = X2.values.reshape(1,-1)
X2 = X2.to_frame()
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_feature = onehot_encoder.fit_transform(X2)
X2 = integer_encoded_feature

print(type(X2))
X = pd.concat([X, X2])

y2 = dataset.iloc[:,8]
y2 = y2.values.reshape(1,-1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_feature = onehot_encoder.fit_transform(y2)
y2 = integer_encoded_feature
y = dataset.iloc[:,9:13]
y["type"] = y2[0]
print(X)
print(y)
# create model
model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='linear'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, y, epochs=50, batch_size=25)
# evaluate the model
scores = model.evaluate(X, y)
print(model.predict(X.iloc[:10]))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))