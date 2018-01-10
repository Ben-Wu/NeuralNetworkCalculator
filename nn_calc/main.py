import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical


def create_model(output_classes):
    model = Sequential()

    model.add(Dense(64, input_dim=2))
    model.add(Dense(64))
    model.add(Dense(output_classes))

    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


def generate_seed():
    return random.randint(0, 123123321)


def generate_data(size, min, max, num_classes, seed=None):
    """Generate random arrays to use as data"""
    np.random.seed(seed or generate_seed())

    x = np.random.randint(low=min, high=max, size=(size, 2))
    y = to_categorical(np.array([e[0] + e[1] for e in x]), num_classes)

    return x, y


def data_generator(size, low, high, num_classes, batch_size=10, seed=None):
    """Generate random arrays to use as data"""
    seed = seed or generate_seed()

    while True:
        np.random.seed(seed)
        for i in range(0, size, batch_size):
            x = np.random.randint(low=low, high=high,
                                  size=(min(batch_size, size - i), 2))
            y = to_categorical(np.array([e[0] + e[1] for e in x]), num_classes)
            yield x, y


if __name__ == '__main__':
    data_size = 7000
    min_input = 0
    max_input = 10
    max_output = max_input * 2

    model = create_model(max_output)
    train_datagen = data_generator(data_size, min_input, max_input,
                                   max_output, seed=123)

    model.fit_generator(train_datagen, data_size, epochs=15, verbose=True)

    test_x, test_y = generate_data(100, min_input, max_input, max_output,
                                   seed=432)

    for x, y in zip(test_x, test_y):
        preds = model.predict(np.array([x]))[0]
        max_prob = np.max(preds)
        print('{} + {} = predicted: {}, actual: {}'
              .format(x[0], x[1],
                      np.where(preds == max_prob)[0][0],
                      np.where(y == 1)[0][0]))
