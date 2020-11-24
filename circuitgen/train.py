import circuitgen
import numpy as np
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
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


def train_features_to_value(features,values,data):
    features = np.transpose(features)
    values = np.transpose(data.values)
    cross_Validation(features,values)
    # print(train_values)
    # test_values = values[81:]
    # model = circuitgen.models.features_to_values()
    # model.fit(train_features, train_values,epochs=1000, batch_size=1)
    # model.summary()
    # for i in range(len(test_features)):
    #     predict = model.predict(np.reshape(test_features[i],(1,3)))
    #     print(test_values[i])
    #     print(predict)

def cross_Validation(input, output):
    kfold = KFold(n_splits=10, shuffle=True)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
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


