import pandas as pd
import inflect

from other.CONSTANTS import *
from other.file_paths import datasets_path, processed_data_path
from source.preprocessing.parse_collection_xml_files import parse_collection
from source.preprocessing.handle_multitype_entities import find_multitype_entities


def load_collection(train):
    file_path = processed_data_path(train) + 'ddi_init.csv'
    try:
        ddi = pd.read_csv(file_path)

    except FileNotFoundError:

        DRUGBANK_PATH, MEDLINE_PATH = datasets_path(train)
        pairs = parse_collection(DRUGBANK_PATH) + parse_collection(MEDLINE_PATH)

        ddi = pd.DataFrame(pairs, columns=pairs[0].keys())
        plural_to_singular(ddi)
        ddi.to_csv(file_path, sep=',', encoding='utf-8')

    # print_stats(ddi)
    return ddi, find_multitype_entities(ddi)


def plural_to_singular(ddi):
    """
    Πχ: penicillin και penicillins αναφέρονται στην ίδια οντότητα
    Αντικαθιστά τον πληθυντικό με ενικό στο ddi
    :param ddi: Το DF με τις αλληλεπιδράσεις
    :return: -
    """
    p = inflect.engine()
    unique_entities = ddi[E1].append(ddi[E2]).unique()  # όλες οι οντότητες τις συλλογής
    print('Unique entities before :', len(unique_entities))
    for entity in unique_entities:
        singular = p.singular_noun(entity)
        if not singular:
            continue
        if singular in unique_entities:  # που είναι στον πληθ, και υπάρχει και ο ενικός τους
            ddi.loc[ddi[E1] == entity, E1] = singular
            ddi.loc[ddi[E2] == entity, E2] = singular

    ddi.sort_values([E1, E2], inplace=True, ignore_index=True)
    print('Unique entities after:', len(ddi[E1].append(ddi[E2]).unique()))



def print_stats(ddi):
    # entities per type count
    print(
        pd.concat([ddi[[E1, E1_TYPE]].rename(columns={E1: 'E', E1_TYPE: 'TYPE'}),
                   ddi[[E2, E2_TYPE]].rename(columns={E2: 'E', E2_TYPE: 'TYPE'})
                   ]).drop_duplicates()
            .groupby(['TYPE']).count()
    )

    # relation per type count
    print(
        ddi[[PAIR_ID, PAIR_TYPE]].groupby([PAIR_TYPE]).count()
    )

    # entity types
    print(
        pd.concat([ddi[[DOC_ID, E1, E1_TYPE]].rename(columns={E1: 'E', E1_TYPE: 'TYPE'}),
                   ddi[[DOC_ID, E2, E2_TYPE]].rename(columns={E2: 'E', E2_TYPE: 'TYPE'})
                   ])
            .groupby(['E', 'TYPE']).count()
    )

