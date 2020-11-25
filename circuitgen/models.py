from keras.models import Sequential

from keras.layers import Input, concatenate, Dense, Dropout
from keras.models import Model
from keras.constraints import NonNeg

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

def features_to_values():
    model = Sequential()
    model.add(Dense(243, input_shape=(3,), activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(81, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(27, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    return model
