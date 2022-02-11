import utils
import numpy as np
import os
import argparse
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
import models
import metrics
from sklearn.metrics import (
    roc_curve,
)

DATA_DIR = "./data/cleaned/"
INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")
OUTPUTS_DIR = "./outputs/"
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results/cv/")

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
        help="Please specify whether you want to run cross validation on CircR2Disease or Circ2Disease dataset ",
        required=True,
    )
    args = parser.parse_args()
    data_name = args.data_name

    # check if data name is valid
    assert data_name in ["CircR2Disease", "Circ2Disease"]
    # load data
    all_pairs = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
    print(len(all_pairs))
    all_labels = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
    [unique_circrnas, unique_diseases] = utils.load_pickle(
        INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
    )

    GIP_CD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
    GIP_DC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
    GIP_DM = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DM.pkl")

    circrna_feature_matrices = [GIP_CD]
    disease_feature_matrices = [GIP_DC, GIP_DM]

    RESULTS = []
    test_scores = []
    roc_curve_data = []
    fold = 1

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in skf.split(all_pairs, all_labels):
        X_train, X_test = all_pairs[train_index], all_pairs[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]
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
        X_test_circrna_vecs = np.array(
            [
                utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
                for c, d in X_test
            ]
        )
        X_test_disease_vecs = np.array(
            [
                utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
                for c, d in X_test
            ]
        )
        X_test = np.concatenate([X_test_circrna_vecs, X_test_disease_vecs], axis=1)

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
            )
        )
        acc, f1, prec, rec, auc, predicted_probas = metrics.evaluate_model(
            dnn, X_test_encoded, y_test.reshape(y_test.shape[0], 1)
        )
        false_positive_rate, true_positive_rate, threshold = roc_curve(
            y_test.reshape(y_test.shape[0], 1), predicted_probas
        )
        roc_curve_data.append(
            (fold, false_positive_rate, true_positive_rate, threshold)
        )
        RESULTS.append([fold, acc, f1, prec, rec, auc])
        print(f"fold: {fold}", acc, f1, prec, rec, auc)
        fold += 1

    test_acc_scores = np.array(
        [acc for (acc, f1, prec, rec, auc, predicted_probas) in test_scores]
    )
    test_f1_scores = np.array(
        [f1 for (acc, f1, prec, rec, auc, predicted_probas) in test_scores]
    )
    test_prec_scores = np.array(
        [prec for (acc, f1, prec, rec, auc, predicted_probas) in test_scores]
    )
    test_rec_scores = np.array(
        [rec for (acc, f1, prec, rec, auc, predicted_probas) in test_scores]
    )
    test_auc_scores = np.array(
        [auc for (acc, f1, prec, rec, auc, predicted_probas) in test_scores]
    )

    RESULTS.append(
        (
            "average",
            f"{test_acc_scores.mean():.3f} +- {test_acc_scores.std():.3f}",
            f"{test_f1_scores.mean():.3f} +- {test_f1_scores.std():.3f}",
            f"{test_prec_scores.mean():.3f} +- {test_prec_scores.std():.3f}",
            f"{test_rec_scores.mean():.3f} +- {test_rec_scores.std():.3f}",
            f"{test_auc_scores.mean():.3f} +- {test_auc_scores.std():.3f}",
        )
    )
    result_file = os.path.join(RESULTS_DIR, f"{data_name}_results.xlsx")
    roc_file = os.path.join(RESULTS_DIR, f"{data_name}_roc_auc_curve.png")
    pd.DataFrame(RESULTS).to_excel(result_file)
    utils.plot_roc_auc_curve(roc_curve_data, roc_file)
    print(
        f"Results are saved to:  {result_file}\nRoc AUC curve plot is saved to: {roc_file}"
    )
