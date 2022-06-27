import json
from os.path import exists

import torch
from torch import tanh
from torch.nn.functional import leaky_relu, relu

from config import PROJECT_PATH
from other.utils import device
from source.classification_task.gnn_models.gnn_model import GNN

MODEL_FILE   = PROJECT_PATH + 'data/models/classification_model.bin'
MODEL_CONFIG_FILE = PROJECT_PATH + 'data/models/classification_model_config.json'


def ClassificationModel(load_model=False):
    with open(MODEL_CONFIG_FILE, "r") as fp:
        model_config = json.load(fp)

    model_config['act_func'] = _resolve_act_func(model_config['act_func'])

    model = GNN(
        # Πάντα ίδια :
        in_channels=model_config['in_channels'],
        hidden_channels=model_config['hidden_channels'],
        node_types=model_config['node_types'],
        edge_types=[tuple(edge) for edge in model_config['edge_types']],

        # Παράμετροι μοντέλου :
        preproc=model_config['preproc'],
        postproc=model_config['postproc'],
        n_conv_layers=model_config['n_conv_layers'],
        conv_type=model_config['conv_type'],

        act_func=model_config['act_func'],

    )
    if load_model:
        if not files_found():
            raise Exception('Saved model file not found. Train the model')

        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
    model.to(device())
    print(model)
    return model


def _resolve_act_func(act_func):
    if act_func == 'tanh':
        return tanh
    elif act_func == 'leaky_relu':
        return leaky_relu
    elif act_func == 'relu':
        return relu


def files_found():
    return exists(MODEL_FILE) and exists(MODEL_CONFIG_FILE)