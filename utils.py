import numpy as np
from math import exp
import random
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm


seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

os.environ["PYTHONHASHSEED"] = "0"

# Create all dirs
Path("data/cleaned").mkdir(parents=True, exist_ok=True)
Path("inputs/features").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)
Path("outputs/results/cv").mkdir(parents=True, exist_ok=True)
Path("outputs/results/predictions").mkdir(parents=True, exist_ok=True)


def save_pickle(dir, filename, var):
    with open(os.path.join(dir, filename), "wb") as f:
        pickle.dump(var, f)


def load_pickle(dir, filename):
    with open(os.path.join(dir, filename), "rb") as f:
        return pickle.load(f)


def interaction_matrix(unique_circrnas, unique_diseases, pairs, labels):
    M = np.zeros((len(unique_circrnas), len(unique_diseases)))
    for ind1, c in enumerate(unique_circrnas):
        for ind2, d in enumerate(unique_diseases):
            if [c, d] in pairs:
                M[ind1, ind2] = labels[pairs.index([c, d])]
    return M


def calculate_width(M):
    width = 0
    for m in range(len(M)):
        width += np.sum(M[m] ** 2) ** 0.5
    width /= len(M)
    return width


def collect_circrna_names(df, columns, delimiter):
    df["all"] = None
    for ind, row in df.iterrows():
        info = []
        for col in columns:
            if isinstance(row[col], str) and (len(row[col]) > 2):
                info += row[col].split(delimiter)
        df.loc[ind, "all"] = "/".join(info)
    return df


def GIP(x1, x2, width):
    return exp((np.sum((x1 - x2) ** 2) ** 0.5 * width) * (-1))


def get_GIP_matrix(values, IM, width):
    values_size = len(values)
    M = np.zeros((values_size, values_size))
    for ind1, v1 in enumerate(values):
        v1_vec = IM[values.index(v1)]
        for ind2, v2 in enumerate(values):
            if v1 == v2:
                M[ind1, ind2] = 1.0
                continue
            v2_vec = IM[values.index(v2)]
            M[ind1, ind2] = GIP(v1_vec, v2_vec, width)
    return M


def get_negatif_samples(sample_size, pos_sample, x, y):
    neg_sample = []
    x_size, y_size = len(x), len(y)
    while len(neg_sample) != sample_size:
        rand_circrna = x[random.randint(0, x_size - 1)]
        rand_disease = y[random.randint(0, y_size - 1)]
        rand_sample = (rand_circrna, rand_disease)
        if (rand_sample not in pos_sample) and (rand_sample not in neg_sample):
            neg_sample.append(rand_sample)
    return neg_sample


def get_circrna_vec(all_circRNAs, feature_matrices, circrna):
    circrna_index = all_circRNAs.index(circrna)
    vecs = []
    for m in feature_matrices:
        vec = m[circrna_index]
        vecs.append(vec)
    return np.concatenate(vecs)


def get_disease_vec(all_diseases, feature_matrices, disease):
    disease_index = all_diseases.index(disease)
    vecs = []
    for m in feature_matrices:
        vec = m[disease_index]
        vecs.append(vec)
    return np.concatenate(vecs)


def plot_roc_auc_curve(roc_curve_data, filename):
    plt.subplots(1, figsize=(10, 10))
    for fold, false_positive_rate, true_positive_rate, threshold in roc_curve_data:
        ek = "th"
        if fold == 2:
            ek = "nd"
        elif fold == 3:
            ek = "rd"
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{fold}{ek} Fold",
            linewidth=3,
        )
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel("True Positive Rate", fontsize=22)
    plt.xlabel("False Positive Rate", fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right", fontsize=18)
    plt.show()
    plt.savefig(filename, format="png")


def sequence_similarity(seq1, seq2):
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    substitute_costs[:] = 2
    return 1 - (
        (lev(seq1, seq2, substitute_costs=substitute_costs)) / (len(seq1) + len(seq2))
    )


def get_sequence_sim_matrix(unique_circrnas, circrna_sequences):
    from weighted_levenshtein import lev
    SIM_CC = np.empty((len(unique_circrnas), len(unique_circrnas)))
    SIM_CC[:] = -1

    for ind1, c1 in tqdm.tqdm(enumerate(unique_circrnas)):
        seq1 = circrna_sequences[c1]
        for ind2, c2 in enumerate(unique_circrnas):
            seq2 = circrna_sequences[c2]
            if (seq1 is None) or (seq2 is None):
                SIM_CC[ind1, ind2] = 0.0
                continue
            if c1 == c2:
                SIM_CC[ind1, ind2] = 1.0
                continue
            if SIM_CC[ind2, ind1] != -1:
                SIM_CC[ind1, ind2] = SIM_CC[ind2, ind1]
                continue
            SIM_CC[ind1, ind2] = sequence_similarity(seq1, seq2)
    return SIM_CC


def generate_dag(trees):
    dag = {}
    for tree in trees:
        diseases = tree.split(".")
        for i in range(len(diseases)):
            disease = ".".join(diseases[: i + 1])
            ind = len(diseases) - i - 1
            dag.setdefault(disease, 0)
            if dag[disease] < ind:
                dag[disease] = ind
    return dag


def DV(dag, coef):
    return sum([DA(dag, disease, coef) for disease in dag])


def DA(dag, disease, coef):
    return coef ** dag[disease]


def semantic_similarity(trees1, trees2, coef=0.8):
    dag1 = generate_dag(trees1)
    dag2 = generate_dag(trees2)
    common_diseases = set(dag1.keys()).intersection(set(dag2.keys()))
    if len(common_diseases) == 0:
        return 0.0
    common_diseases_score = sum(
        [
            DA(dag1, disease, coef) + DA(dag2, disease, coef)
            for disease in common_diseases
        ]
    )
    overall_score = DV(dag1, coef) + DV(dag2, coef)
    return common_diseases_score / overall_score


def get_semantic_similarity_matrix(unique_diseases, disease_mesh_tree_dict):
    SIM_DD = np.empty((len(unique_diseases), len(unique_diseases)))
    SIM_DD[:] = -1

    for ind1, d1 in tqdm.tqdm(enumerate(unique_diseases)):
        d1_trees = disease_mesh_tree_dict[d1] if d1 in disease_mesh_tree_dict else None
        for ind2, d2 in enumerate(unique_diseases):
            if d1 == d2:
                SIM_DD[ind1, ind2] = 1.0
                continue
            if SIM_DD[ind2, ind1] != -1:
                SIM_DD[ind1, ind2] = SIM_DD[ind2, ind1]
                continue
            d2_trees = (
                disease_mesh_tree_dict[d2] if d2 in disease_mesh_tree_dict else None
            )
            if (d1_trees is None) or (d2_trees is None):
                SIM_DD[ind1, ind2] = 0.0
                continue
            SIM_DD[ind1, ind2] = semantic_similarity(d1_trees, d2_trees)
    return SIM_DD
