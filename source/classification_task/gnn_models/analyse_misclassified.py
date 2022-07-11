import pickle
from os.path import exists

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, pyplot

from other.CONSTANTS import *
from config import PROJECT_PATH
from source.database_pcg.n4j_pcg.query_neo4j import Query_Neo4j as Neo4j

MISCLASSIFIED = PROJECT_PATH + 'data/models/misclassified.csv'


def run_miscls(errors, print_results=False):
    if not print_results:
        return
    try:
        n4j = Neo4j(train=False)
        errors = errors.to_dict('records')

        query = f'''
        unwind $errors as pair
        match (d1:{DRUG_CLASS})-[:{INT_FOUND}]->(i:{INTERACTION_CLASS})<-[:{INT_FOUND}]-(d2:{DRUG_CLASS})
        match (i)-[:{SENT_SOURCE}]->(s:{SENT_CLASS})-[:{SENT_TEXT}]->(text:{DATA_NODE})
        match (t2:{OWL_CLASS})<-[:{RDF_TYPE}]-(i2:{INTERACTION_CLASS})-[:{SENT_CON_INTER}]-(s)
        match (d3:{DRUG_CLASS})-[:{INT_FOUND}]->(i2)<-[:{INT_FOUND}]-(d4:{DRUG_CLASS})
        where i.key = pair.{INTERACTION_CLASS} and d1.key < d2.key and d3.key < d4.key
        return pair.{INTERACTION_CLASS}, d1.key, d2.key, pair.y, pair.y_hat, text.key, collect([d3.key, d4.key, t2.key])
        '''
        result = n4j.session.run(query=query, errors=errors).values()
        result = pd.DataFrame(result, columns=['Interaction', 'd1', 'd2', 'y', 'y_hat', 'text', 'pairs'])
        result['collection'] = [p.split('.', 1)[0] for p in result[INTERACTION_CLASS].values]
        result.to_csv(MISCLASSIFIED, sep=',', encoding='utf-8', index=False)
        print('Misclassified results file:', MISCLASSIFIED)

        print_(result, n4j)
    except Exception:
        return


def print_(result, n4j):
    pd.options.mode.chained_assignment = None
    print('\n', '='*50)
    print('# unique sentences =', len(result['text'].unique()))
    print('Miscls per collection:\n', result[['collection', INTERACTION_CLASS]].groupby(['collection']).count())
    ipairs = n4j.run_query(f'''match (i:{INTERACTION_CLASS}) return i.key''')
    ipairs = [p[0].split('.', 1)[0] for p in ipairs]

    ipairs = pd.DataFrame(ipairs, columns=['i'])
    ipairs['n'] = [1 for i in range(ipairs.shape[0])]
    print('Total pairs per collection:\n',ipairs.groupby(['i']).count())
    print('\n', '=' * 50)
    # ============================================================================================================
    if not exists(PROJECT_PATH + 'data/models/sentences.pkl'):
        from source.classification_task.dataset_preparation.bert_dataset import Bert_Dataset, SENTENCE_EMB
        sentences_init = n4j.run_query(n4j.q_collect_sentences())
        Bert_Dataset(sentences_init=sentences_init, embeddings_mode=SENTENCE_EMB, run_for_bert=False)

    tokens = pickle.load(open(f"{PROJECT_PATH}data/models/sentences.pkl", "rb"))
    false_negative = result[result['y_hat'] == 'negative']
    false_negative['sentence'] = [p[:p.rfind('.')] for p in false_negative[INTERACTION_CLASS].values]
    false_negative['pairs'] = [len(p) for p in false_negative['pairs'].values]
    false_negative = false_negative[['sentence', 'd1', 'd2', 'y', 'text', 'pairs']]
    false_negative['len'] = [len(tokens[sid]) for sid in false_negative['sentence'].values]

    sentences = false_negative[['sentence', 'y', 'pairs', 'len', 'text']]
    sentences = sentences.groupby(['sentence', 'len', 'pairs', 'text']).size().reset_index(name='counts').sort_values(
        ['len', 'counts'])
    print(sentences[:5], '...', sentences[-5:], sep='\n')

    numeric = sentences[['len', 'pairs', 'counts']]
    numeric.rename(columns={'len': 'Μήκος πρότασης', 'pairs': '# ζευγών', 'counts': '# FN'}, inplace=True)
    print(numeric.describe())
    # ===============================================================================================================

    pyplot.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, 120, 10.0))
    sns.boxplot(y="variable", x="value", data=pd.melt(numeric),  width=.6)
    plt.show()

    # ================================================================================================================

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), squeeze=True)
    n = list(range(numeric.shape[0]))
    marker_sentlen, stemline_sentlen, _ = ax.stem(n, numeric['Μήκος πρότασης'].values)
    marker_fn, stemline_fn, _           = ax.stem(n, numeric['# FN'].values)
    marker_pairs, stemline_pairs, _     = ax.stem(n, numeric['# ζευγών'].values)

    # stem and lines plot options
    plt.setp(marker_sentlen, color='#7CAE00', markersize=6, markeredgewidth=2, label='Μήκος πρότασης')
    plt.setp(stemline_sentlen, color='black')
    plt.setp(marker_fn, color='r', markersize=6, markeredgewidth=2, label='# FN')
    plt.setp(stemline_fn, color='black')
    plt.setp(marker_pairs, color='#00BFC4', markersize=6, markeredgewidth=2, label='# ζευγών')
    plt.setp(stemline_pairs, color='black')

    ax.grid(axis='y')
    plt.legend(numpoints=1, fontsize=9)
    plt.show()















