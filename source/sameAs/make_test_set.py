from other.file_paths import SYNONYMS_DATA
from source.sameAs.make_sameAs_pairs import *

file = SYNONYMS_DATA + 'similarity_labeled_set.csv'


def make_set(ddi):
    """
    Δημιουργία του test set για την αξιολόγηση των μετρικών που αναπτύχθηκαν
    προκειμένου να εντοπιστούν ζεύγη ονομάτων sameAs.
    """
    synonyms = Synonyms()
    sameAsObj = SameAs(fz_type=RATIO, fz_cutoff=70)
    df = sameAsObj.find_similar_pairs(ddi[E1].append(ddi[E2]).unique().tolist())

    for i in df.index:
        w1 = df[E1][i]
        w2 = df[E2][i]

        if synonyms.check_if_synonyms(w1, w2):
            df.at[i, LABEL] = 1
            df.at[i, ANNOTATED] = 'synonyms'
            continue  # βρέθηκε στα συνώνυμα της βάσης, συνέχισε στο επόμενο pair

        if sameAsObj.different_formulas(w1, w2):
            # έχουν διαφορετικό μοριακό τύπο
            df.at[i, LABEL] = 0
            df.at[i, ANNOTATED] = 'pubchem_formula'

    df.to_csv(file, sep=',', encoding='utf-8', index=False)



def format_manually_annotated():
    df = pd.read_csv(file)
    # 2982 before -> 2608 after
    df = df[df[LABEL] != -1].reset_index(drop=True)
    df.loc[df[LABEL] == True, LABEL] = 1
    df.loc[df[LABEL] == False, LABEL] = 0
    df.to_csv(file, sep=',', encoding='utf-8', index=False)



def test_set_stats():
    df = pd.read_csv(file)[[LABEL, ANNOTATED]]

    # Positive / negative examples
    print(df.groupby([LABEL]).count())

    # annotation source
    print(df.groupby(ANNOTATED).count())

    # Positive / Negative per source
    print(df.groupby([LABEL, ANNOTATED]).size().reset_index(name='counts'))

