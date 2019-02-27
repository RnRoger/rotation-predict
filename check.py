# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder


def convert_to_int(aa):
    d = {'CYS': '1', 'ASP': '8', 'SER': '9', 'GLN': '16', 'LYS': '17',
        'ILE': '2', 'PRO': '7', 'THR': '10', 'PHE': '15', 'ASN': '18', 
        'GLY': '3', 'HIS': '6', 'LEU': '11', 'ARG': '14', 'TRP': '19', 
        'ALA': '4', 'VAL':'5', 'GLU': '12', 'TYR': '13', 'MET': '20'}
    
    return int(d[aa])

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv("rotamers_han_test.csv")
dataset = dataset.dropna()
# split into input (X) and output (Y) variables
X = dataset.iloc[:,[3,4,5,6]]
dataset['residue_type'] = dataset['residue_type'].apply(convert_to_int)
X2 = dataset.iloc[:,8]


X["type"] = X2[0]
print(X)
y = dataset.iloc[:,9:13]
y["type"] = X2[0]

# create model
model = load_model('model.h5')

# evaluate the model
scores = model.evaluate(X, y)
print(model.predict(X.iloc[:10]))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
