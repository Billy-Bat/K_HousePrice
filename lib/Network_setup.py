from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def load_NN(input_shape, Output_shape, layers, n_units=100) :
    model = Sequential()

    model.add(Dense(input_shape, kernel_initializer='normal', input_dim = input_shape, activation='relu'))
    for i in range(layers):
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(Output_shape, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model
