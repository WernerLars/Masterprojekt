from tensorflow import keras
from tensorflow.keras.layers import *


def createMLP():
    model = keras.Sequential()
    model.add(Input(shape=(36,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model



def createMLP2376():
    model = keras.Sequential()
    model.add(Input(shape=(2376,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model