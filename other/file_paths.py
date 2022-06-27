from config import PROJECT_PATH, BERT_MODEL_FOLDER
# ==================================================================================

# raw collection files
init_data_folder = PROJECT_PATH + 'data/ddi'

def datasets_path(train):
    folder = PROJECT_PATH + 'data/ddi/' + \
             ('train/' if train else
              'test/extraction_task/')
    return folder + 'DrugBank', folder + 'MedLine'
# ----------------------------------------------------------------------------------
# collection after processing
processed_data_folder = PROJECT_PATH + 'data/processed_data'

def processed_data_path(train):
    return PROJECT_PATH + 'data/processed_data/' + ('train/' if train else 'test/')

# ==================================================================================

# sameAs related files
SYNONYMS_DATA = PROJECT_PATH + 'data/synonyms_data/'

drugbank_vocab_file = SYNONYMS_DATA + 'drugbank vocabulary.csv'
synonyms_db_file = SYNONYMS_DATA + 'synonyms.csv'
formulas_db_file = SYNONYMS_DATA + 'formulas.db'
test1 = SYNONYMS_DATA + 'similarity_labeled_set.csv'
test2 = SYNONYMS_DATA + 'ismp_confused_drug_names.csv'

synomyms_files_list = [drugbank_vocab_file, synonyms_db_file, formulas_db_file, test1, test2]
# ==================================================================================

# input dataset for cls task
def graph_dataset(train):
    return PROJECT_PATH + 'data/graph_dataset/' + ('train' if train else 'test')

# bert model path
bert_path = PROJECT_PATH + f'data/models/bert/{BERT_MODEL_FOLDER}/'


