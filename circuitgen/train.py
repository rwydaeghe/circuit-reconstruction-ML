import circuitgen
import numpy as np


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




