import utils
import numpy as np
import os
import argparse
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
import models


INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")
OUTPUTS_DIR = "./outputs/"
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")

epochs = 1000
batch_size = 32
run_ae = True
encoding_dim = 512
layer_sizes = (1024, 512, 256, 128, 64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        help="Please specify whether you want to train CircR2Disease or Circ2Disease dataset ",
        required=True,
    )
    args = parser.parse_args()
    data_name = args.data_name

    # check if data name is valid
    assert data_name in ["CircR2Disease", "Circ2Disease"]
    # load data
    X_train = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
    y_train = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
    [unique_circrnas, unique_diseases] = utils.load_pickle(
        INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
    )

    GIP_CD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
    GIP_DC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
    GIP_DM = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DM.pkl")

    circrna_feature_matrices = [GIP_CD]
    disease_feature_matrices = [GIP_DC, GIP_DM]

    X_train_circrna_vecs = np.array(
        [
            utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
            for c, d in X_train
        ]
    )
    X_train_disease_vecs = np.array(
        [
            utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
            for c, d in X_train
        ]
    )
    X_train = np.concatenate([X_train_circrna_vecs, X_train_disease_vecs], axis=1)
    ae_input_dim = X_train.shape[1]

    for tr_ind, val_ind in StratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=0
    ).split(X_train, y_train):
        X_train_, X_val = X_train[tr_ind], X_train[val_ind]
        y_train_, y_val = y_train[tr_ind], y_train[val_ind]

    if run_ae:
        autoencoder, encoder = models.train_autoencoder(
            X_train_,
            X_val,
            X_train_.shape[1],
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size,
        )

        X_train_encoded = encoder.predict(X_train)
        X_train_encoded_ = encoder.predict(X_train_)
        X_val_encoded = encoder.predict(X_val)
    else:
        autoencoder, ae_input_dim = None, None
        X_train_encoded = X_train
        X_train_encoded_ = X_train_
        X_val_encoded = X_val
        encoding_dim = X_train_encoded.shape[1]
    autoencoder.save(os.path.join(MODELS_DIR, f"{data_name}_autoencoder.h5"))
    encoder.save(os.path.join(MODELS_DIR, f"{data_name}_encoder.h5"))

    dnn = models.train_dnn_model(
        X_train_encoded_,
        y_train_.reshape(y_train_.shape[0], 1),
        X_val_encoded,
        y_val.reshape(y_val.shape[0], 1),
        input_dim=encoding_dim,
        layer_sizes=layer_sizes,
        epochs=epochs,
        batch_size=batch_size,
    )
    dnn.save(os.path.join(MODELS_DIR, f"{data_name}_dnn.h5"))
