from cross_validation import RESULTS_DIR
import utils
import numpy as np
import os
import argparse
from keras.models import load_model
import pandas as pd


INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")
OUTPUTS_DIR = "./outputs/"
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

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
        help="Please specify whether you want to use model trained with CircR2Disease or Circ2Disease dataset",
        required=True,
    )
    parser.add_argument(
        "--disease_name",
        type=str,
        help="Please specify the disease name that you want to find most likely associated circRNAs",
        required=True,
    )
    args = parser.parse_args()
    data_name = args.data_name
    disease_name = args.disease_name

    assert data_name in ["CircR2Disease", "Circ2Disease"]

    # load data
    all_pairs = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
    all_labels = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
    [unique_circrnas, unique_diseases] = utils.load_pickle(
        INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
    )
    unique_diseases_str = "\n".join(unique_diseases)

    assert (
        disease_name in unique_diseases
    ), f"Disease name is not valid. Pick one of the following diseases: {unique_diseases_str}"

    GIP_CD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
    GIP_DC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
    GIP_DM = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DM.pkl")

    # load models
    autoencoder = load_model(os.path.join(MODELS_DIR, f"{data_name}_autoencoder.h5"))
    encoder = load_model(os.path.join(MODELS_DIR, f"{data_name}_encoder.h5"))
    dnn = load_model(os.path.join(MODELS_DIR, f"{data_name}_dnn.h5"))

    # create possible novel disease-circRNA pairs
    pairs_df = pd.DataFrame(all_pairs, columns=["circRNA", "disease"])
    pairs_df["label"] = all_labels
    pos_samples = pairs_df[pairs_df.label == 1].drop("label", axis=1).values.tolist()

    candidate_pairs = np.array(
        [
            [circRNA, disease_name]
            for circRNA in unique_circrnas
            if [circRNA, disease_name] not in pos_samples
        ]
    )

    circrna_feature_matrices = [GIP_CD]
    disease_feature_matrices = [GIP_DC, GIP_DM]
    X = candidate_pairs

    X_circrna_vecs = np.array(
        [
            utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
            for c, d in X
        ]
    )
    X_disease_vecs = np.array(
        [
            utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
            for c, d in X
        ]
    )
    X_pair = np.concatenate([X_circrna_vecs, X_disease_vecs], axis=1)
    ae_input_dim = X_pair.shape[1]
    X_pair_encoded = encoder.predict(X_pair)

    candidate_pairs_df = pd.DataFrame(candidate_pairs)
    candidate_pairs_df["pred"] = dnn.predict(X_pair_encoded).round(5)
    candidate_pairs_df["pred_class"] = dnn.predict_classes(X_pair_encoded)
    candidate_pairs_df.columns = [
        "CircRNA",
        "Disease",
        "Association Score",
        "Association Prediction",
    ]
    candidate_pairs_df = candidate_pairs_df.sort_values(
        "Association Score", ascending=False
    )
    candidate_pairs_df.to_excel(
        os.path.join(
            RESULTS_DIR, f"{disease_name}_{data_name}_predicted_novel_pairs.xlsx"
        )
    )
