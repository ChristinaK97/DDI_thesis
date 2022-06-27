import os
import random
from os import mkdir
from os.path import exists
import numpy as np
import patoolib
import torch
import warnings
import torch_geometric
from config import *
from other.file_downloader import download_files
from other.file_paths import *
# ===================================================================================

def check_folders():
    if not exists(PROJECT_PATH + '/data'):
        mkdir(PROJECT_PATH + '/data')

    check_collection()
    check_neo4j()
    check_models()

    return check_sameAs_files()

def check_collection():
    if not exists(processed_data_folder):

        if not exists(init_data_folder):
            download_files(init_data_folder + '.rar')
            patoolib.extract_archive(init_data_folder + '.rar', outdir=PROJECT_PATH + 'data')
            os.remove(init_data_folder + '.rar')

        if not exists(init_data_folder):
            raise Exception(f'Neither raw or processed data files found. Failed to download raw.'
                            f'Raw collection files can be manually downloaded from \n'
                            f'https://github.com/ChristinaK97/DDI_thesis_files/blob/main/data/ddi.rar \n'
                            f'After downloading extract ddi.rar file to PROJECT_PATH/data (as set in the config.py file):\n'
                            f'{PROJECT_PATH}/data')
        else:
            for subfolder in ['', '/train', '/test']:
                mkdir(processed_data_folder + subfolder)


def check_neo4j():
    if not exists(NEO4J_PATH[:-1]):
        raise Exception(f'\nNeo4j folder {NEO4J_PATH[:-1]} not found. The community version 4.0.0 can be downloaded from \n'
                        f'https://drive.google.com/file/d/1tZwJasXVJvlXWMn5mo3SsgwplOaruyYi/view?usp=sharing .\n'
                        f'After downloading extract neo4j.rar file to (as set in the config.py file): \n'
                        f"{NEO4J_PATH[:NEO4J_PATH.find('/')]}/")

    if not exists(NEO4J_PATH + 'import'):
        mkdir(NEO4J_PATH + 'import')


def check_sameAs_files():

    run_make_db = False

    if not exists(SYNONYMS_DATA[:-1]):
        mkdir(SYNONYMS_DATA[:-1])
        download_files(synomyms_files_list)

    if RESET_SYNONYMS_DB:
        for file in [synonyms_db_file, formulas_db_file]:
            if exists(file):
                os.remove(file)

    if not exists(synonyms_db_file):
        to_download = []
        if not exists(drugbank_vocab_file):
            to_download.append(drugbank_vocab_file)
        if not RESET_SYNONYMS_DB:
            to_download.append(synonyms_db_file)

        download_files(to_download)

        if not exists(synonyms_db_file):
            run_make_db = True
        if not exists(drugbank_vocab_file):
            warnings.warn(drugbank_vocab_file + ' not found.')

    if not exists(formulas_db_file):
        if not RESET_SYNONYMS_DB:
            download_files(formulas_db_file)
        if not exists(formulas_db_file):
            run_make_db = True

    return run_make_db


def check_models():
    if not exists(PROJECT_PATH + '/data/models'):
        mkdir(PROJECT_PATH + '/data/models')
        mkdir(PROJECT_PATH + '/data/models/bert')
        raise_bert_warning('No saved pretrained model found.')


def raise_bert_warning(message):
    warnings.warn(
        f'\n{message} A BERT model must be downloaded or an exception might be raised during training.\n'
        'BioBERT can be downloaded from \n'
        'https://drive.google.com/file/d/1egCaVAGqlXsgleqQzgLnKhfQJALDzwvi/view?usp=sharing \n'
        'After downloading extract biobert_v1.1_pubmed.rar file to '
        f'{PROJECT_PATH}data/models/bert'
        '\n (as set in the config.py file).\n'
        'Alternatively, a BERT model (pytorch_model.bin, config.json, vocab.txt) can be downloaded from HuggingFace \n'
        f'and placed in {PROJECT_PATH}data/models/bert/BERT_MODEL_FOLDER.\n'
        f'BERT_MODEL_FOLDER value can be set in config.py. (Current value:{BERT_MODEL_FOLDER})')

# ===================================================================================
def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch_geometric.seed_everything(0)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


