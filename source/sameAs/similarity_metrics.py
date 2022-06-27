from math import sqrt
import distance
from thefuzz import fuzz as fz

COSINE = 1
JACCARD = 2
LEVEN = 3
FZ = 5

RATIO = 1
PARTIAL_R = 2
TOKEN_SORT_R = 3
TOKEN_SET_R = 4


def calc_similarity(w1, w2, sim_metric, fz_type=None):
    if sim_metric == COSINE:
        return cos_sim(w1, w2)
    elif sim_metric == JACCARD:
        return jaccard_sim(w1, w2)
    elif sim_metric == LEVEN:
        return levenshtein(w1, w2)
    elif fz_type is not None:
        return fuzzy_sim(w1, w2, fz_type=fz_type)
    else:
        raise Exception('Invalid arguments')


def jaccard_sim (w1, w2):

    w1_letters = set(w1)
    w2_letters = set(w2)
    intersection = w1_letters.intersection(w2_letters) if len(w1) > len(w2) else \
                   w2_letters.intersection(w1_letters)

    j_sim = len(intersection) / len(w1_letters.union(w2_letters))
    return j_sim


def word_vector(word):
    counter = dict.fromkeys(word, 0)
    for l in word: counter[l] += 1

    return counter


def cos_sim(w1, w2):
    def vector_length(v):
        return sum(letter_count ** 2 for letter_count in v.values())

    v1 = word_vector(w1)
    l1 = vector_length(v1)
    v2 = word_vector(w2)
    l2 = vector_length(v2)

    intersection = set(v1.keys()).intersection(v2.keys()) if len(w1) > len(w2) else \
        set(v2.keys()).intersection(v1.keys())
    try:
        return sum(v1[c] * v2[c] for c in intersection) \
               / sqrt(l1 * l2)

    except ZeroDivisionError:
        return 0


def levenshtein(w1, w2):
    return distance.levenshtein(w1, w2)


def fuzzy_sim (w1, w2, fz_type):

    if fz_type == RATIO:
        return fz.ratio(w1, w2)
    elif fz_type == PARTIAL_R:
        return fz.partial_ratio(w1, w2)
    elif fz_type == TOKEN_SORT_R:
        return fz.token_sort_ratio(w1, w2)
    elif fz_type == TOKEN_SET_R:
        return fz.token_set_ratio(w1, w2)
    else:
        raise Exception('Invalid type')


