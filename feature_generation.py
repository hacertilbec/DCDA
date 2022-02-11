import pandas as pd
import utils
import numpy as np
import os
import argparse
from sklearn.utils import shuffle

DATA_DIR = "./data/cleaned/"
INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")

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
    data_df = pd.read_csv(
        os.path.join(DATA_DIR, f"{data_name}.csv"), sep="*", index_col=0
    )

    if "disease" in data_df.columns:
        data_df["Disease Name"] = data_df["Disease Name"].fillna(data_df["disease"])
    data_df["Disease Name"].dropna(inplace=True)
    data_df = data_df.drop_duplicates(subset=["circRNA Name", "Disease Name"])

    unique_circrnas = data_df["circRNA Name"].unique().tolist()
    unique_diseases = data_df["Disease Name"].unique().tolist()
    """
    pos_sample = list(zip(data_df["circRNA Name"], data_df["Disease Name"]))
    pos_sample_size = len(pos_sample)
    neg_sample = utils.get_negatif_samples(
        pos_sample_size, pos_sample, unique_circrnas, unique_diseases
    )

    all_pairs = np.array(pos_sample + neg_sample)
    all_labels = np.array([1] * len(pos_sample) + [0] * len(neg_sample))
    all_pairs, all_labels = shuffle(all_pairs, all_labels, random_state=0)

    # ------- GIP_CD and GIP_DC -------
    # circRNA - disease interaction
    M_cd_train = utils.interaction_matrix(
        unique_circrnas, unique_diseases, all_pairs.tolist(), all_labels.tolist()
    )
    cd_width_train = utils.calculate_width(M_cd_train)

    # disease - circRNA interaction
    M_dc_train = M_cd_train.T
    dc_width_train = utils.calculate_width(M_dc_train)

    GIP_CD = utils.get_GIP_matrix(unique_circrnas, M_cd_train, cd_width_train)
    GIP_DC = utils.get_GIP_matrix(unique_diseases, M_dc_train, dc_width_train)

    # ------- GIP_DM -------
    HMDD = pd.read_csv("data/cleaned/HMDD.csv", sep="*", index_col=0)
    HMDD = HMDD[HMDD["Disease Name"].isin(unique_diseases)]
    disease_mir_pairs = list(map(list, zip(HMDD["Disease Name"], HMDD["mir"])))
    HMDD_diseases = HMDD["Disease Name"].unique().tolist()
    HMDD_mirnas = HMDD["mir"].unique().tolist()

    M_dm = utils.interaction_matrix(
        unique_diseases, HMDD_mirnas, disease_mir_pairs, [1] * len(disease_mir_pairs)
    )
    dm_width = utils.calculate_width(M_dm)
    GIP_DM = utils.get_GIP_matrix(unique_diseases, M_dm, dm_width)

    # ------- SAVE INPUT DATA -------
    utils.save_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl", all_pairs)
    utils.save_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl", all_labels)
    utils.save_pickle(
        INPUT_DIR,
        f"{data_name}_unique_circrnas_diseases.pkl",
        [unique_circrnas, unique_diseases],
    )

    utils.save_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl", GIP_CD)
    utils.save_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl", GIP_DC)
    utils.save_pickle(FEATURES_DIR, f"{data_name}_GIP_DM.pkl", GIP_DM)"""

    if data_name == "CircR2Disease":
        # ------- GIP_CM -------
        circbank = pd.read_csv("data/cleaned/circBank.csv", sep="*", index_col=0)

        circ_mir_pairs = list(
            map(list, zip(circbank["circRNA Name"], circbank["miR_ID"]))
        )
        circbank_circrnas = circbank["circRNA Name"].unique().tolist()
        circbank_mirnas = circbank["miR_ID"].unique().tolist()

        M_cm = utils.interaction_matrix(
            unique_circrnas, circbank_mirnas, circ_mir_pairs, [1] * len(circ_mir_pairs)
        )
        cm_width = utils.calculate_width(M_cm)
        GIP_CM = utils.get_GIP_matrix(unique_circrnas, M_cm, cm_width)

        # ------- SIM_CC -------
        circ_seq_dict = utils.load_pickle(INPUT_DIR, "circrna_sequence_dict.pkl")
        data_df["Sequence"] = data_df["circRNA Name"].replace(circ_seq_dict)
        circrna_sequences = data_df[["circRNA Name", "Sequence"]].drop_duplicates()
        circrna_sequences = dict(
            zip(circrna_sequences["circRNA Name"], circrna_sequences["Sequence"])
        )
        SIM_CC = utils.get_sequence_sim_matrix(unique_circrnas, circrna_sequences)

        # ------- SIM_DD -------
        disease_mesh_tree_dict = utils.load_pickle(
            INPUT_DIR, "disease_mesh_tree_dict.pkl"
        )
        SIM_DD = utils.get_semantic_similarity_matrix(
            unique_diseases, disease_mesh_tree_dict
        )

        utils.save_pickle(FEATURES_DIR, f"{data_name}_GIP_CM.pkl", GIP_CM)
        utils.save_pickle(FEATURES_DIR, f"{data_name}_SIM_CC.pkl", SIM_CC)
        utils.save_pickle(FEATURES_DIR, f"{data_name}_SIM_DD.pkl", SIM_DD)
