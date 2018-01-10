import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def create_model():
    model = Sequential()

    model.add(Dense(64, input_dim=2))
    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    return model


def generate_data(size, min, max, seed=None):
    """Generate random arrays to use as data"""
    if not seed:
        random.seed(seed)

    x = np.random.randint(low=min, high=max, size=(size, 2))
    y = np.array([e[0] + e[1] for e in x])

    return x, y


if __name__ == '__main__':
    pass
