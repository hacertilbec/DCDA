import utils
import numpy as np
import os
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
import models
import metrics
import pandas as pd
from itertools import permutations
from tqdm import tqdm

DATA_DIR = "./data/cleaned/"
INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")
OUTPUTS_DIR = "./outputs/"
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results/cv/")


epochs = 1000
batch_size = 32
run_ae = True
data_name = "CircR2Disease"

if __name__ == "__main__":
    RESULTS = []
    # load data
    all_pairs = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
    print(len(all_pairs))
    all_labels = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
    [unique_circrnas, unique_diseases] = utils.load_pickle(
        INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
    )
    print(len(all_pairs))
    print(len(unique_circrnas), len(unique_diseases))

    GIP_CD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
    GIP_DC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
    GIP_DM = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DM.pkl")
    SIM_DD = utils.load_pickle(FEATURES_DIR, f"{data_name}_SIM_DD.pkl")

    features_dict = {
        "GIP_CD": GIP_CD,
        "GIP_DC": GIP_DC,
        "GIP_DM": GIP_DM,
        "SIM_DD": SIM_DD,
    }
    circrna_feature_matrices_names = ["GIP_CD"]
    disease_feature_matrices_names = ["GIP_DC", "GIP_DM"]
    for circrna_feature_matrices_names, disease_feature_matrices_names in [
        [["GIP_CD"], ["GIP_DC", "SIM_DD"]],
        [["GIP_CD"], ["GIP_DC", "GIP_DM", "SIM_DD"]],
    ]:
        for encoding_dim in [128, 256, 512, 1024]:
            for layer_sizes in [
                (256, 512, 256, 128, 64),
                (512, 512, 256, 128, 64),
                (1024, 512, 256, 128, 64),
            ]:
                circrna_feature_matrices = [
                    features_dict[i] for i in circrna_feature_matrices_names
                ]
                disease_feature_matrices = [
                    features_dict[i] for i in disease_feature_matrices_names
                ]
                test_scores = []
                fold = 1

                skf = StratifiedKFold(n_splits=5, shuffle=False)
                for train_index, test_index in skf.split(all_pairs, all_labels):
                    X_train, X_test = all_pairs[train_index], all_pairs[test_index]
                    y_train, y_test = all_labels[train_index], all_labels[test_index]
                    X_train_circrna_vecs = np.array(
                        [
                            utils.get_circrna_vec(
                                unique_circrnas, circrna_feature_matrices, c
                            )
                            for c, d in X_train
                        ]
                    )
                    X_train_disease_vecs = np.array(
                        [
                            utils.get_disease_vec(
                                unique_diseases, disease_feature_matrices, d
                            )
                            for c, d in X_train
                        ]
                    )
                    X_train = np.concatenate(
                        [X_train_circrna_vecs, X_train_disease_vecs], axis=1
                    )
                    ae_input_dim = X_train.shape[1]
                    X_test_circrna_vecs = np.array(
                        [
                            utils.get_circrna_vec(
                                unique_circrnas, circrna_feature_matrices, c
                            )
                            for c, d in X_test
                        ]
                    )
                    X_test_disease_vecs = np.array(
                        [
                            utils.get_disease_vec(
                                unique_diseases, disease_feature_matrices, d
                            )
                            for c, d in X_test
                        ]
                    )
                    X_test = np.concatenate(
                        [X_test_circrna_vecs, X_test_disease_vecs], axis=1
                    )

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
                        X_test_encoded = encoder.predict(X_test)
                    else:
                        autoencoder, ae_input_dim = None, None
                        X_train_encoded = X_train
                        X_train_encoded_ = X_train_
                        X_val_encoded = X_val
                        X_test_encoded = X_test
                        encoding_dim = X_train_encoded.shape[1]

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

                    test_scores.append(
                        metrics.evaluate_model(
                            dnn, X_test_encoded, y_test.reshape(y_test.shape[0], 1)
                        )[:-1]
                    )
                    fold += 1

                test_acc_scores = np.array(
                    [acc for (acc, f1, prec, rec, auc) in test_scores]
                )
                test_f1_scores = np.array(
                    [f1 for (acc, f1, prec, rec, auc) in test_scores]
                )
                test_prec_scores = np.array(
                    [prec for (acc, f1, prec, rec, auc) in test_scores]
                )
                test_rec_scores = np.array(
                    [rec for (acc, f1, prec, rec, auc) in test_scores]
                )
                test_auc_scores = np.array(
                    [auc for (acc, f1, prec, rec, auc) in test_scores]
                )

                RESULTS.append(
                    (
                        ", ".join(circrna_feature_matrices_names),
                        ", ".join(disease_feature_matrices_names),
                        encoding_dim,
                        str(layer_sizes),
                        f"{test_acc_scores.mean():.4f} +- {test_acc_scores.std():.4f}",
                        f"{test_f1_scores.mean():.4f} +- {test_f1_scores.std():.4f}",
                        f"{test_prec_scores.mean():.4f} +- {test_prec_scores.std():.4f}",
                        f"{test_rec_scores.mean():.4f} +- {test_rec_scores.std():.4f}",
                        f"{test_auc_scores.mean():.4f} +- {test_auc_scores.std():.4f}",
                    )
                )
                print(
                    ", ".join(circrna_feature_matrices_names),
                    ", ".join(disease_feature_matrices_names),
                    encoding_dim,
                    str(layer_sizes),
                    f"{test_acc_scores.mean():.4f} +- {test_acc_scores.std():.4f}",
                    f"{test_f1_scores.mean():.4f} +- {test_f1_scores.std():.4f}",
                    f"{test_prec_scores.mean():.4f} +- {test_prec_scores.std():.4f}",
                    f"{test_rec_scores.mean():.4f} +- {test_rec_scores.std():.4f}",
                    f"{test_auc_scores.mean():.4f} +- {test_auc_scores.std():.4f}",
                )

    pd.DataFrame(RESULTS).to_excel(os.path.join(RESULTS_DIR, "model_optimization.xlsx"))
