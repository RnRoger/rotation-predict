# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import random
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

def getdata():
    # load dataset
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
    return X, y


def create_model(X, y, hidden_layer_nodes, loss, optimizer, activation, epoch, seed, layer):
    # create model
    numpy.random.seed(seed) # 3 = 0.7
    model = Sequential()
    model.add(Dense(5, input_dim=5, activation='relu'))
    for _ in range(layer):
        model.add(Dense(hidden_layer_nodes, activation=activation))
    model.add(Dense(5, activation='linear'))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(X, y, epochs=epoch, batch_size=20)
    return model


def evalmodel(X, y, model):
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


def main():
    # hidden_layer_nodes_list = [2,3,4,6,8,10]
    # loss_list = ["mse", "binary_crossentropy"]
    # optimizer_list = ["adam", "sgd", "rmsprop"]
    # activation_list = ["relu", "linear"]
    # epoch_list = [5]
    # seed_list = [7]
    hidden_layer_nodes_list = [3,5,8,12,50,100,300]
    layer_list = [1,2,3]
    loss_list = ["mse"]
    optimizer_list = ["rmsprop", "adam"]
    activation_list = ["relu", "linear"]
    epoch_list = [4]

    total_tests = len(hidden_layer_nodes_list)*len(loss_list)*len(optimizer_list)*len(activation_list)*len(epoch_list)*len(layer_list)
    count = 1
    cont = True
    with open("results.csv", "w") as results:
        results.write("hidden_layer_nodes,loss,optimizer,activation,epoch,seed,layer,correct\n")
        X, y = getdata()
        while cont == True:
            total_tests = "infinity"
            for hidden_layer_nodes in hidden_layer_nodes_list:
                for loss in loss_list:
                    for optimizer in optimizer_list:
                        for activation in activation_list:
                            for epoch in epoch_list:
                                for layer in layer_list:
                                    seed = random.randint(1,(2**32) -1)
                                    # 1422780283 = 0.8
                                    model = create_model(X, y, hidden_layer_nodes, loss, optimizer, activation, epoch, seed, layer)
                                    correct = evalmodel(X, y, model)
                                    print(f"Test {count} of {total_tests} has an accuracy of {correct}")
                                    if correct >= 0.70:
                                        results.write(f"{hidden_layer_nodes},{loss},{optimizer},{activation},{epoch},{seed},{layer},{correct}\n")
                                        print(f"{hidden_layer_nodes},{loss},{optimizer},{activation},{epoch},{seed},{layer},{correct}\n")
                                    count += 1
                                    if correct >= 0.95:
                                        cont = False

    model.save("model.h5")

if __name__ == "__main__":
    main()