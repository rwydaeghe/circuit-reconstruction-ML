from keras.models import Sequential

from keras.layers import Input, concatenate, Dense
from keras.models import Model


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

