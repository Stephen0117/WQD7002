import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class cnn_mel:

    def Lenet5():
        model = models.Sequential()
        model.add(layers.Conv2D(28, kernel_size=(5, 5),strides=(1, 1), activation='tanh',padding='same', input_shape=(256, 256, 3)))
        model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))
        model.add(layers.Conv2D(10, (5, 5),strides=(1, 1), activation='tanh'))
        model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), activation='tanh'))
        model.add(layers.Flatten())
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def VGG16():
        model = models.Sequential()
        model.add(layers.Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=1, activation="sigmoid"))

        return model

    def Alexnet():
        model = models.Sequential()
        model.add(layers.Conv2D(55,kernel_size=(11,11),padding="valid",strides=4,activation="relu",input_shape=(256,256,3)))
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=2,padding="valid"))
        model.add(layers.Conv2D(27,kernel_size=(5,5),padding="same",strides=1,activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=2,padding="valid"))
        model.add(layers.Conv2D(13,kernel_size=(3,3),padding="same",strides=1,activation="relu"))
        model.add(layers.Conv2D(13,kernel_size=(3,3),padding="same",strides=1,activation="relu"))
        model.add(layers.Conv2D(13,kernel_size=(3,3),padding="same",strides=1,activation="relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(units=1, activation="sigmoid"))

        return model
