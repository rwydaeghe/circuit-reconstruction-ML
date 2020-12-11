from keras.models import Sequential
from keras.layers import Input, concatenate, Dense, Dropout
from keras.models import Model
from keras.constraints import NonNeg
from tensorflow.keras.backend import mean, square
from tensorflow.keras.optimizers import Adam,Adadelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

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
    model.add(Dense(5, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))

    model.add(Dense(20, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_constraint=NonNeg()))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='linear', kernel_constraint=NonNeg()))

    opt = Adadelta(lr=0.001)
    model.compile(loss='mae', optimizer=opt,metrics=["accuracy"])
    return model


def linear():
    return LinearRegression()


def lasso():
    return Lasso(alpha=0.0001, precompute=True, max_iter=1000,positive=True, random_state=9999, selection='random')

def svr():
    return  SVR(kernel='poly', C=1e3, degree=2)


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
    model.add(Dense(20,activation='relu', kernel_constraint=NonNeg()))
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
