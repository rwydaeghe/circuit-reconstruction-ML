from keras.models import Sequential

from keras.layers import Input, concatenate, Dense, Dropout
from keras.models import Model
from keras.constraints import NonNeg
from tensorflow.python.keras.backend import mean, square

def regression_model(size):

    model_netlist = Sequential()
    model_netlist.add(Dense(20, input_shape=(None, 3), kernel_initializer='he_uniform', activation='relu'))
    model_netlist.add(Dense(size))

    model_analyses = Sequential()
    model_analyses.add(Dense(20, input_size=16, kernel_initializer='he_uniform', activation='relu'))
    model_analyses.add(Dense(size))

    combined_input = concatenate(model_netlist.output, model_analyses.output)
    x = Dense(size)(combined_input)

    model = Model(inputs=[model_netlist.input, model_analyses.input], outputs=x)
    model.compile(loss='mae', optimizer='adam')

    return model

def mlp():
    model = Sequential()
    model.add(Dense(20, input_shape=(3,), activation='relu', kernel_constraint=NonNeg()))
    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dense(3, activation='linear', kernel_constraint=NonNeg()))
    model.compile(loss='mae', optimizer='adam',metrics=["accuracy"])
    return model


def regression_chain_start():
    model = Sequential()
    model.add(Dense(20, input_shape=(3,),activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(5,activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear', kernel_constraint=NonNeg()))
    model.compile(loss='mae', optimizer='adam')
    return model


def regression_chain_middle():
    model = Sequential()
    model.add(Dense(20, input_shape=(4,),activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(5,activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear', kernel_constraint=NonNeg()))
    model.compile(loss='mae', optimizer='adam')
    return model


def regression_chain_end():
    model = Sequential()
    model.add(Dense(20, input_shape=(5,),activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear', kernel_constraint=NonNeg()))
    model.compile(loss='mae', optimizer='adam')
    return model


def regression_loss_function(y_predicted, y_actual):

    custom_loss_value = mean(square((y_actual-y_predicted))*10)
    return custom_loss_value
