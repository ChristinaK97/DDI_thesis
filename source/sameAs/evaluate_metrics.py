import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as accuracy, \
    precision_score as precision, \
    recall_score as recall

from source.sameAs.similarity_metrics import calc_similarity as dist
from source.sameAs.similarity_metrics import *
from source.sameAs.make_sameAs_pairs import SameAs
from other.CONSTANTS import LABEL, e, E2, E1
from other.file_paths import test1, test2

pd.set_option('display.width', 500)
pd.set_option("display.max_rows", None, "display.max_columns", None)


def run_evaluation(set_path=test1, multicrit=False):

    test_set = pd.read_csv(set_path)
    pairs = test_set[[E1, E2]]
    labels = test_set[LABEL].values
    if multicrit:
        multicrit_evaluation(pairs, labels)
    else:
        single_crit_evaluation(pairs, labels, dist_metric=JACCARD, fz_type=None)


def multicrit_evaluation(pairs, labels):
    #sameAs = SameAs(fz_type=TOKEN_SORT_R, fz_cutoff=85, fz_thrs=94, sec_cr=COSINE, sec_thrs=0.965, reset_synonyms=True) \
    #    .get_sameAs_triples(pairs, rename=False)
    s1 = SameAs(fz_type=RATIO,        fz_cutoff=85, fz_thrs=95, sec_cr=COSINE, sec_thrs=0.965, reset_synonyms=True)
    s2 = SameAs(fz_type=TOKEN_SORT_R, fz_cutoff=85, fz_thrs=94, sec_cr=COSINE, sec_thrs=0.965, reset_synonyms=False)
    sameAs = s1.join(pairs, s2, rename=False)

    multicrit_eval(pairs, labels, sameAs)


def multicrit_eval(pairs, labels, sameAs):

    y_pred = np.zeros(pairs.shape[0])

    for _, pair in sameAs.iterrows():
        i = pairs[
                ((pairs[E1] == pair[E1]) & (pairs[E2] == pair[E2])) |
                ((pairs[E2] == pair[E1]) & (pairs[E1] == pair[E2]))]
        if i.shape[0] != 0:
            i = i.index.values.astype(int)[0]
            y_pred[i] = pair['y_pred']
    sameAs.sort_values(by=['sim'], ignore_index=True, ascending=False, inplace=True)
    print(sameAs)
    results = classification_metrics(labels, y_pred)
    print(results)



def single_crit_evaluation(pairs, labels, dist_metric, fz_type=None):

    dist_vector = get_distances_vector(pairs, dist_metric=dist_metric, fz_type=fz_type)
    print(dist_vector[:10])

    acc, prec, rec, count = ([] for _ in range(4))
    range_ = np.arange(0.70, 1, 0.01)
    if fz_type is not None:
        range_ *= 100

    for thrs in range_:
        y_pred = predict(dist_vector, thrs=thrs)
        results = classification_metrics(labels, y_pred)

        for l, arg in [(acc, 0), (prec, 1), (rec, 2), (count, 3)]:
            l.append(results[arg])

    plot(range_, ['Precision', 'Recall', 'Accuracy'], prec, rec, acc)
    plot(range_, ['Count'], count)


def get_distances_vector(pairs, dist_metric, fz_type=None):
    return [
        dist(pair[E1], pair[E2], sim_metric=dist_metric, fz_type=fz_type)
        for _, pair in pairs.iterrows()]


def predict(dist_vector, thrs):
    y_pred = [
        1. if pair_dist + e >= thrs else 0. for pair_dist in dist_vector]
    return np.array(y_pred)



def classification_metrics(labels, y_pred):
    return \
        accuracy(y_true=labels, y_pred=y_pred), \
        precision(y_true=labels, y_pred=y_pred, pos_label=1., zero_division=0), \
        recall(y_true=labels, y_pred=y_pred, pos_label=1., zero_division=0), \
        np.count_nonzero(y_pred)


def plot(range_, plot_labels, *results):
    color = ['tab:cyan', 'tab:orange', 'tab:green']
    plt.style.use(plt.style.library['seaborn-whitegrid'])
    plt.figure(figsize=(5, 5))
    # plt.ylim([-0.05, 1.05])

    plt.title("metric - Test set x")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    
    for i, metric in enumerate(results):
        plt.plot(list(range_), metric, color = color[i])

    plt.legend(plot_labels, loc='lower left')
    plt.show()


run_evaluation()

