import torch
from torch import nn
from torch.nn.functional import dropout
from torch.nn.functional import softmax
from torch.nn.modules.container import ModuleDict
from torch_geometric.nn import Linear
from torch_geometric.nn import SAGEConv, GINConv, GATConv, GCNConv, HeteroConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn import BatchNorm

from other.CONSTANTS import INTERACTION_CLASSES, INTERACTION_CLASS



class GNN_Model(nn.Module):

    # hidden_channels -> final GNN layer node embedding (len(z_v))
    def __init__(self, hidden_channels):

        nn.Module.__init__(self)

        # prediction head
        self.W = Linear(in_channels=hidden_channels, out_channels=len(INTERACTION_CLASSES))

    @staticmethod
    def ConvLayer(lazy_init, conv_type, hidden_channels, act_func):

        if conv_type == 'SAGEConv':
            return SAGEConv(in_channels=lazy_init, out_channels=hidden_channels, normalize=True)
        elif conv_type == 'GCNConv':
            return GCNConv(in_channels=lazy_init, out_channels=hidden_channels, normalize=True)
        elif conv_type == 'GINConv':
            return GINConv(MLP(in_channels=-1, hidden_channels=hidden_channels, out_channels=hidden_channels,
                               num_layers=3, act=act_func.__name__, dropout=0.4))
        elif conv_type == 'GATConv':
            return GATConv(in_channels=lazy_init, out_channels=hidden_channels, dropout=0.1)

    def forward(self, data, nodes_to_predict):
        x = data.x_dict
        edge_index = data.edge_index_dict

        z = self.encoder(x, edge_index)

        y_hat = self.prediction_head(z, nodes_to_predict)

        return y_hat

    def prediction_head(self, z, nodes_to_predict):

        y_hat = self.W(z[nodes_to_predict])

        return y_hat if self.training else self.get_predictions(y_hat)

    def get_predictions(self, y_hat):
        y_hat = softmax(y_hat, dim=1)
        return torch.argmax(y_hat, dim=1), y_hat


# =========================================================================================


class GNN_with_PrePost(GNN_Model):

    def __init__(self, in_channels, hidden_channels,
                 preproc=None, postproc=None):

        final_emb_dim = hidden_channels if postproc is None else postproc[-1]

        GNN_Model.__init__(self, final_emb_dim)

        self.pre_linear = self.MLP_Net(in_channels, preproc, self.node_types)
        self.post_linear = self.MLP_Net(hidden_channels, postproc, [INTERACTION_CLASS])

    def encoder(self, x, edge_index):

        if self.pre_linear is not None:
            x = {node_type: self.pre_linear[node_type](x[node_type])
                 for node_type in self.pre_linear}

        x = self.conv_layers(x, edge_index)

        if self.post_linear is not None:
            x = {node_type: self.post_linear[node_type](x[node_type])
                 for node_type in self.post_linear}

        return x[INTERACTION_CLASS]

    def MLP_Net(self, in_channels, hidden_layers, node_types):

        if hidden_layers is None:
            return None
        mlp = ModuleDict({
            node_type: MLP([in_channels] + hidden_layers, act=self.act_func.__name__, dropout=0.4)
            for node_type in node_types
        })
        return mlp


# ===================================================================================================


class GNN(GNN_with_PrePost):

    def __init__(self, in_channels, hidden_channels,
                 node_types, edge_types,
                 n_conv_layers, conv_type='SAGEConv',
                 preproc=None, postproc=None, act_func=None
                 ):
        self.node_types = node_types
        self.n_conv_layers = n_conv_layers
        self.act_func = act_func

        GNN_with_PrePost.__init__(self, in_channels, hidden_channels,
                                  preproc=preproc, postproc=postproc)

        # conv layers
        self.convs = nn.ModuleList()

        for _ in range(self.n_conv_layers):
            conv = self.RConvLayer(conv_type, hidden_channels, edge_types)
            self.convs.append(conv)

        """self.norm = nn.ModuleList([
            BatchNorm(in_channels=hidden_channels)
            for _ in range(self.n_conv_layers)
        ])"""

    def RConvLayer(self, conv_type, hidden_channels, edge_types, aggr='sum'):

        conv = {}
        for edge_type in edge_types:
            lazy_init = -1 if edge_type[0] == edge_type[2] else (-1, -1)
            conv[edge_type] = GNN_Model.ConvLayer(lazy_init, conv_type, hidden_channels, self.act_func)

        conv = HeteroConv(conv, aggr=aggr)
        return conv

    def conv_layers(self, x, edge_index):

        for i in range(self.n_conv_layers):
            x = self.convs[i](x, edge_index)
            # x[INTERACTION_CLASS] = self.norm[i](x[INTERACTION_CLASS])

            x = {node_type: dropout(x_i, p=0.4, training=self.training) for node_type, x_i in x.items()}

            if i < self.n_conv_layers - 1:
                x = self.act(x)

            return x

    def act(self, x):
        return {node_type: self.act_func(x_i) for node_type, x_i in x.items()} \
            if self.act_func is not None else x






