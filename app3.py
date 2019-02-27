# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder

# Split train set into train and test
# Keras' accuracy doesn't make sense, calculate your own accuracy by comparing predicted with desired scores

def convert_to_int(aa):
    d = {'CYS': '1', 'ASP': '8', 'SER': '9', 'GLN': '16', 'LYS': '17',
        'ILE': '2', 'PRO': '7', 'THR': '10', 'PHE': '15', 'ASN': '18', 
        'GLY': '3', 'HIS': '6', 'LEU': '11', 'ARG': '14', 'TRP': '19', 
        'ALA': '4', 'VAL':'5', 'GLU': '12', 'TYR': '13', 'MET': '20'}
    
    return int(d[aa])

def getscore(X, y):
    # evaluate the model
    correctly_identified = 0
    scores = model.evaluate(X, y)
    print(y.iloc[:10])
    print(model.predict(X.iloc[:10]).round(0).astype('int'))
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    correct = y.to_numpy()
    predicted = model.predict(X)
    correct = correct[:5]
    # predicted = predicted.round(0).astype('int')
    predicted = predicted[:5].round(0).astype('int')
    print(correct)
    # print(predicted2)
    print(predicted)
    for row_nr in range(len(correct)):
        for column_nr in range(5):
            # print(correct[row_nr][column_nr],">",predicted[row_nr][column_nr])
            if correct[row_nr][column_nr] == predicted[row_nr][column_nr]:
                correctly_identified += 1
    return correctly_identified/(len(correct)*5)
# fix random seed for reproducibility
numpy.random.seed(1422780283) # 3
# load pima indians dataset
dataset = pd.read_csv("rotamers_han_train.csv")
dataset = dataset.fillna(0)
# split into input (X) and output (Y) variables
X = dataset.iloc[:,[3,4,5,6]]
dataset['residue_type'] = dataset['residue_type'].apply(convert_to_int)
X2 = dataset.iloc[:,8]


X["type"] = X2[0]
print(X)
y = dataset.iloc[:,9:13]
y["type"] = X2[0]
print(y)
# create model
model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(5, activation='linear'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, y, epochs=4, batch_size=20)
# evaluate the model
scores = model.evaluate(X, y)
print(y.iloc[:10])
print(model.predict(X.iloc[:10]).round(0).astype('int'))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(getscore(X, y))
model.save("model.h5")