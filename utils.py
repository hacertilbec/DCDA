import numpy as np
from math import exp
import random
import pickle
import os
import matplotlib.pyplot as plt


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
        if rand_sample not in pos_sample:
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
    plt.savefig(filename)
