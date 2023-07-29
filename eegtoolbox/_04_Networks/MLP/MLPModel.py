from keras import Input
from keras.layers import Dense
from tensorflow import keras


def createMLP(input):
    model = keras.Sequential()
    model.add(Input(shape=(input,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model