import pandas as pd

from other.CONSTANTS import *
from config import NEO4J_RESET, MODE
from other.file_paths import SYNONYMS_DATA
from other.utils import check_folders
from source.database_pcg.formulas_db import FormulasDB
from source.database_pcg.n4j_pcg.query_neo4j import Query_Neo4j
from source.database_pcg.synonyms_db import Synonyms
from source.preprocessing.negative_filtering import NegativeFiltering
from source.preprocessing.preprocessing import load_collection
from source.preprocessing.handle_multitype_entities import multitype_sameAs
from source.preprocessing.generate_rdf_triples import make_rdf_triples
from source.database_pcg.n4j_pcg.initialize_neo4j import Initialize_Neo4j
from source.classification_task.gnn_models.train_classification_model import \
    TrainClassificationModel, INFERENCE, TRAINING


# =======================================================================================

def run_pipeline(train):
    pd.set_option('display.width', 500)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # ------------------------------------------------------------------------------------
    print('=' * 50, '\n', 'TRAIN' if train else 'TEST', ' SET', sep='')
    # load collection
    print("Load collection")
    ddi, multitype_entities = load_collection(train)

    # find same as triples
    print('Find sameAs pairs')
    from source.sameAs.make_sameAs_pairs import load_sameAs
    sameAs = multitype_sameAs(load_sameAs(ddi, train), ddi, multitype_entities)

    # make rdf triples
    print("Make RDF Dataframe")
    triples = make_rdf_triples(train, ddi, set(multitype_entities.keys()), sameAs)

    if NEO4J_RESET:
        # write to neo4j
        print("Write to Neo4j")
        Initialize_Neo4j(train, triples=triples)
        # remove filtered pairs and isolated nodes
        print('Negative Instance Filtering :')
        NegativeFiltering(train=train)
        Query_Neo4j(train=train).new_pairs_from_sameAs()


def make_db():
    # make synonyms / formulas db
    unique_entities = set()
    for train in [False, True]:
        ddi, _ = load_collection(train)
        unique_entities = unique_entities.union(set(ddi[E1].append(ddi[E2]).unique()))

    test2 = SYNONYMS_DATA + 'ismp_confused_drug_names.csv'
    evaluation_set = pd.read_csv(test2)
    evaluation_set = set(evaluation_set[E1].append(evaluation_set[E2]).unique())
    unique_entities = unique_entities.union(evaluation_set)

    print("Make synomyms db:")
    Synonyms(entities=unique_entities)
    print("Make formulas db:")
    FormulasDB(entities=unique_entities)


def main():
    run_make_db = check_folders()
    if run_make_db :
        make_db()
    for train in [False, True]:
        run_pipeline(train=train)

    TrainClassificationModel(mode=INFERENCE if MODE == 'INFERENCE' else TRAINING)


if __name__ == "__main__":
    main()








