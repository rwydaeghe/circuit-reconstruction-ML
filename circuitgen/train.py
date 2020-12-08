from tensorflow.python.keras.backend import mean, square

import circuitgen
import numpy as np
import torch
from torch_geometric.data import Data
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import LinearRegression

def train(dirpath):
    input_netlist, output = circuitgen.data.read_netlist(dirpath)
    input_analyses = circuitgen.data.read_transiant_analyses(dirpath)

    cutoff = int(len(input_netlist)/100*80)
    train_netlist = input_netlist[:cutoff]
    test_netlist = input_netlist[cutoff+1:]
    train_analyses = input_analyses[:cutoff]
    test_analyses = input_analyses[cutoff + 1:]
    train_output = output[:cutoff]
    test_output = output[cutoff+1:]

    model = circuitgen.models.regression_model(len(output))

    model.fit(
        x= [train_netlist, train_analyses], y=train_output,
        validation_data=([test_netlist, test_analyses], test_output),
        epochs=50, batch_size=8
    )

    prediction = model.predict[test_netlist, test_analyses]

    difference = prediction.flatten() - test_output
    percent_diff = (difference / test_output) * 100
    abs_percent_diff = np.abs(percent_diff)

    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
    return


def train_features_to_value(features,data):
    features = np.transpose(features)
    values = np.transpose(data.values)
    #cross_Validation(features,values)
    # print(train_values)
    train_features = features[:80]
    train_values = values[:80]
    test_features = features[:80]
    train_values[:, 1] *= 1000
    train_values[:, 2] *= 1000000

    test_values = values[81:]
    model = circuitgen.models.features_to_values()
    model.fit(train_features, train_values,epochs=300, batch_size=1)
    model.summary()
    for i in range(len(test_features)):
        predict = model.predict(np.reshape(test_features[i],(1,3)))
        print(test_values[i])
        print(predict)


def regression_chain(features, data):
    features = np.transpose(features)
    train_features = features[:90]
    test_features = features[91:]
    values = np.transpose(data.values)
    values_start = values[:, 0]
    values_middle = values[:, 1]
    values_end =  values[:, 2]
    train_values_start = values_start[:90]
    test_values_start = values[91:]
    model_start = circuitgen.models.regression_chain_start()
    model_start.fit(train_features, train_values_start, epochs=100,batch_size=1)
    features_middle = np.array([np.append(features[x],values[x][0]) for x in range(len(features))])
    model_middle = circuitgen.models.regression_chain_middle()
    model_middle.fit(features_middle, values_middle, epochs=100, batch_size=1)
    features_end = np.array([np.append(features_middle[x], values[x][1]) for x in range(len(features_middle))])
    model_end = circuitgen.models.regression_chain_end()
    model_end.fit(features_end, values_end, epochs=100, batch_size=1)


def cross_Validation(input, output):
    kfold = KFold(n_splits=10, shuffle=True)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    output[:, 1] *= 1000
    output[:, 2] *= 1000000
    for train, test in kfold.split(input, output):

        model = circuitgen.models.features_to_values()
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        history = model.fit(input[train], output[train],batch_size=1, epochs=300, verbose=1)
        scores = model.evaluate(input[test], output[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')



