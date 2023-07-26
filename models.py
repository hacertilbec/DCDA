from tensorflow.keras import initializers
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import random
from tensorflow.random import set_seed 
from numpy.random import seed
import os
import tensorflow.compat.v1 as tf
import keras

seed_value = 0
random.seed(seed_value)
set_seed(seed_value)
seed(seed_value)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def build_autoencoder(input_dim, encoding_dim):
    K.clear_session()
    # This is our input image
    input_img = keras.Input(shape=(input_dim,))
    hidden_layer = layers.Dense(
        int((encoding_dim + input_dim) / 2),
        activation="relu",
        kernel_initializer=initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=seed_value
        ),
        bias_initializer=initializers.Zeros(),
    )(input_img)
    encoded = layers.Dense(
        encoding_dim,
        activation="relu",
        kernel_initializer=initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=seed_value
        ),
        bias_initializer=initializers.Zeros(),
    )(hidden_layer)
    hidden_layer2 = layers.Dense(
        int((encoding_dim + input_dim) / 2),
        activation="relu",
        kernel_initializer=initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=seed_value
        ),
        bias_initializer=initializers.Zeros(),
    )(encoded)
    decoded = layers.Dense(
        input_dim,
        activation="sigmoid",
        kernel_initializer=initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=seed_value
        ),
        bias_initializer=initializers.Zeros(),
    )(hidden_layer2)

    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    autoencoder.compile(loss="mean_squared_error", optimizer="adam")
    return autoencoder, encoder


def train_autoencoder(X, X_val, input_dim, encoding_dim, epochs=10000, batch_size=32):
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    early_stopping = EarlyStopping(monitor="val_loss", patience=20)
    hist = autoencoder.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=0,
    )
    optimal_epochs = hist.epoch[-1] + 1
    X = np.concatenate([X, X_val])
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(
        X,
        X,
        epochs=optimal_epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )

    return autoencoder, encoder


def build_dnn(input_dim, layer_sizes):
    K.clear_session()
    model = Sequential()
    model.add(
        Dense(
            layer_sizes[0],
            input_dim=input_dim,
            activation="relu",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )
    model.add(
        Dense(
            layer_sizes[1],
            activation="relu",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )
    model.add(
        Dense(
            layer_sizes[2],
            activation="relu",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )
    model.add(
        Dense(
            layer_sizes[3],
            activation="relu",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )
    model.add(
        Dense(
            layer_sizes[4],
            activation="relu",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )
    model.add(
        Dense(
            1,
            activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_value
            ),
            bias_initializer=initializers.Zeros(),
        )
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_dnn_model(
    X,
    y,
    X_val,
    y_val,
    input_dim,
    layer_sizes,
    epochs=10000,
    batch_size=32,
):
    model = build_dnn(input_dim, layer_sizes)

    early_stopping = EarlyStopping(monitor="val_loss", patience=20)
    hist = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )
    optimal_epochs = hist.epoch[-1] + 1

    X = np.concatenate([X, X_val])
    y = np.concatenate([y, y_val])

    model = build_dnn(input_dim, layer_sizes)
    model.fit(
        X,
        y,
        epochs=optimal_epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )
    return model
