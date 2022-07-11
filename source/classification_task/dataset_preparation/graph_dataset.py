import pickle
from os import mkdir
from os.path import exists

import torch
from other.file_paths import graph_dataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import HGTLoader
from torch_geometric.transforms import RandomNodeSplit

from other.CONSTANTS import *
from source.database_pcg.n4j_pcg.query_neo4j import Query_Neo4j as Neo4j
from source.classification_task.dataset_preparation.bert_model_transformers import WordEmbeddings

ZERO_INIT = 0


class GraphDataset(Dataset):

    def __init__(self, train, add_inverse=False):
        self.train = train
        self.f = self.set_files()
        self.label_enc = LabelEncoder().fit(list(INTERACTION_CLASSES))
        self.negative_label = self.label_enc.transform(['negative'])[0]

        super(GraphDataset, self).__init__(root=graph_dataset(self.train), transform=None)

        if add_inverse:
            self.add_inverse_edges()


# -----------------------------------------------------------------------------------------------------------------------

    def set_files(self):
        files = {}
        path = graph_dataset(self.train)
        files['processed_path'] = path + '/processed'
        files['saved_dataset'] = f"{files['processed_path']}/dataset.pt"
        files['dataset_info'] = f"{files['processed_path']}/dataset_info.data"
        files['files'] = [files['saved_dataset'], files['dataset_info']]

        return files

    @property
    def raw_file_names(self):
        """
        Τα αρχεία που πρέπει να είναι αποθηκευμένα ώστε να μη χρειαστεί να τα φτιάξει
        """
        return self.f['files']

    def download(self):
        """
        Εκτελείται αν δε βρήκε τα αρχεία.
        1. Αποθήκευση των subgraph, nodes_per_class, node_labels σε ομώνημα αρχεία
        2. Κατεβάζει τα sentences ώστε να φτιαχτούν τα word embeddings μέσω του biobert.
        """
        n4j = Neo4j(self.train)
        nodes_per_class, subgraph = n4j.graph_representation_triples()
        node_labels = n4j.get_interaction_nodes_labels(self.label_enc)

        sentences_init = n4j.run_query(n4j.q_collect_sentences())  # 2
        WordEmbeddings(train=self.train, sentences_init=sentences_init).word_embeddings()

        n4j.close()
        self.graph, self.n_interaction_nodes, self.node_indexes, self.emb_dim = \
            self.create_graph(subgraph, node_labels, nodes_per_class)
        self.save_dataset()


    def save_dataset(self):
        if not exists(self.f['processed_path']):
            mkdir(self.f['processed_path'])
        torch.save(self.graph, self.f['saved_dataset'])
        dataset_info = {'n_interaction_nodes': self.n_interaction_nodes,
                        'node_indexes': self.node_indexes,
                        'emb_dim': self.emb_dim}
        with open(self.f['dataset_info'], 'wb') as filehandle:
            pickle.dump(dataset_info, filehandle)

    @property
    def processed_file_names(self):
        return 'run_process'

    def process(self):
        self.graph = torch.load(self.f['saved_dataset'])
        dataset_info = pickle.load(open(self.f['dataset_info'], 'rb'))
        self.n_interaction_nodes = dataset_info['n_interaction_nodes']
        self.node_indexes = dataset_info['node_indexes']
        self.emb_dim = dataset_info['emb_dim']


    # ---------------------------------------------------------------------------------------------------------------------

    def create_graph(self, subgraph, node_labels, nodes_per_class):
        node_indexes, node_feature_vectors, edge_index_matrices, node_labels, emb_dim = \
            self.create_vectors(subgraph, node_labels, nodes_per_class)

        graph = HeteroData()
        for node_type, feature_vector in node_feature_vectors.items():
            graph[node_type].x = feature_vector

        graph[INTERACTION_CLASS].y = node_labels
        n_interaction_nodes = graph[INTERACTION_CLASS].y.size(0)

        self.initialize_interaction_nodes_feature_vec(graph=graph, emb_dim=emb_dim)

        graph[INTERACTION_CLASS].train_mask = torch.ones(n_interaction_nodes, dtype=torch.bool)

        for edge_type, edge_index in edge_index_matrices.items():
            graph[edge_type].edge_index = edge_index

        return graph, n_interaction_nodes, node_indexes, emb_dim



    def initialize_interaction_nodes_feature_vec(self, graph=None, emb_dim=None, method=ZERO_INIT):
        if graph is None: graph = self.graph
        if emb_dim is None: emb_dim = self.emb_dim

        n_nodes = graph[INTERACTION_CLASS].y.size(0)
        if method == ZERO_INIT:
            feauture_vector = torch.zeros(size=(n_nodes, emb_dim), dtype=torch.float32)
        else:
            raise Exception('Invalid init method')

        graph[INTERACTION_CLASS].x = feauture_vector



    def create_vectors(self, subgraph, node_labels, nodes_per_class):
        """
        :param subgraph : DF<RDF_S, RDF_P, RDF_O> με τα κατηγορήματα (ακμές που θα μπουν στο γράφο)
        :param node_labels : Ο τύπος κάθε κόμβου interaction, κωδικοποιημένος με το Label Encoder
                             Dict{key = '_:p{i}', value = Label_Encoder(interaction type του i)
        :param nodes_per_class : Οι κορυφές του γράφου οργανωμένες σύμφωνα με τον τύπο τους
                Dict{key   = Ο τύπος/κλάση της κορυφής στο γράφο (Sentence,    Token,           Interaction),
                     value = List[όλοι οι κόμβοι αυτού του τύπου (*.d{x}.s{y}, *.d{x}.s{y}.e{z}, _:p)
        :return: Τους πίνακες χαρακτηριστικών των οντοτήτων, γειτνίασης και ετικετών των interation nodes

        1. node_indexes :
           Dict{ key   = Sentence, Token, Interaction
                 value = Dict<key = οντότητα (όνομα), value = id οντότητας -> με την μορφή Α/Α}}
        2. Για τις κορυφές Sentence και Token, δημιουργεί τα αρχικά διανύσματα χαρακτηριστικών
           όπως προέκυψαν από το bert.
           node_feature_vectors : Dict{ key = Sentence, Token
                value = Tensor (float32) διάστασης πλήθος κορυφών x διάσταση διανύσματος χαρακτηριστικών}
        3. Tensor διάστασης : πλήθος κόμβων interactions με τα labels τους
        4. Δημιουργεί τους πίνακες γειτνίασης για κάθε τύπο ακμής.
           Dict με key = τύπος ακμής ως τριπλέτα (Τύπος κορυφής subject, predicate, Τυπος κορυφής object)
                   value Tensor (long) διάστασης 2 x πλήθος ακμών του συγκεκριμένου τύπου μέσα στο γράφημα.
        """
        node_indexes = self.nodes_to_indexes(nodes_per_class)  # 1

        # LOAD BERT EMBEDDINGS FILES
        sentence_embeddings, token_embeddings = WordEmbeddings(train=self.train).word_embeddings()
        emb_dim = next(iter(sentence_embeddings.values())).size(dim=0)

        # CREATE NODE FEATURE VECTORS
        node_feature_vectors = {}  # 2
        for node_type, embeddings in [(SENT_CLASS, sentence_embeddings), (TOKEN_CLASS, token_embeddings)]:
            node_feature_vectors[node_type] = self.create_node_feature_vectors(
                nodes_per_class[node_type], embeddings, emb_dim
            )

        # CREATE INTERACTION NODE LABELS
        node_labels = self.create_node_labels(nodes_per_class[INTERACTION_CLASS], node_labels,  # 3
                                              node_indexes[INTERACTION_CLASS])

        # CREATE EDGE INDEX MATRICES
        edge_index = self.create_edge_index_matrices(subgraph, node_indexes)  # 4

        return node_indexes, node_feature_vectors, edge_index, node_labels, emb_dim



    def nodes_to_indexes(self, nodes_per_class):
        """
        Αντιστοιχίζει κάθε οντότητα (κόμβο του γράφου) με έναν Α/Α που θα την αντιπροσωπεύει στον πίνακα
        χαρακτηριστικών και γειτνίασης.
        :param nodes_per_class: Οι κορυφές του γράφου οργανωμένες σύμφωνα με τον τύπο τους
                Dict{key   = Ο τύπος/κλάση της κορυφής στο γράφο (Sentence,    Token,           Interaction),
                     value = List[όλοι οι κόμβοι αυτού του τύπου (*.d{x}.s{y}, *.d{x}.s{y}.e{z}, _:p)
        :return: node_indexes :
           Dict{ key   = Sentence, Token, Interaction
                 value = Dict<key = οντότητα (όνομα), value = id οντότητας -> με την μορφή Α/Α}}
        """
        for node_type in nodes_per_class:
            nodes_per_class[node_type] = \
                {node: i for i, node in enumerate(nodes_per_class[node_type])}
        return nodes_per_class



    def create_node_feature_vectors(self, nodes, embeddings, emb_dim):
        """
        Δημιουργεί τον πίνακα των διανυσμάτων χαρακτηριστικών ενός τύπου κορυφής
        του γραφήματος.
        :param emb_dim: Διάσταση των word embeddings (base bert model = 768)
        :param nodes :  List[όλοι οι κόμβοι αυτού του τύπου (Sentence : *.d{x}.s{y} ή Token : *.d{x}.s{y}.e{z})]
        :param embeddings: Dict{key = ο identifier του κόμβου, όπως φαίνεται στο nodes,
                                value = bert embedding αυτής της πρότασης ή του token}
        :return: Tensor (float32) διάστασης πλήθος κορυφών x διάσταση διανύσματος χαρακτηριστικών
                 Προκύπτει ως
                 node_features[node_index] = word_embedding(οντότητα node_index)
                 όπου node_index το id (Α/Α) της οντότητας
        """
        # Πλήθος των κορυφών (οντοτήτων) του γράφου
        n_nodes = len(nodes)

        node_features = torch.empty((n_nodes, emb_dim), dtype=torch.float32)
        for node, index in nodes.items():
            node_features[index] = embeddings[node]

        return node_features



    def create_edge_index_matrices(self, subgraph, node_indexes):
        """
        Δημιουργεί τους πίνακες με τις ακμές του γράφου.
        :return:
        - edge_index_matrices : Dict με key = τύπος ακμής ως τριπλέτα (Τύπος κορυφής subject, predicate, Τυπος κορυφής object)
                    value Tensor (long) διάστασης 2 x πλήθος ακμών του συγκεκριμένου τύπου μέσα στο γράφημα.
                    Οι ακμές αναπαριστάνεται με τη μορφή λίστας συντεταγμένων (COO format) άρα
                    value[0] και value[1] : ids των αρχικών (οντότητα subject)
                    και τελικών (οντότητα object) κορυφών αντίστοιχα
        """
        edge_index_matrices = {}

        for edge_type in subgraph[RDF_P].unique():
            domain, range = get_domain_range(edge_type)
            triples = subgraph[(subgraph[RDF_P] == edge_type)]

            sub_index = [
                node_indexes[domain][s] for s in triples[RDF_S].values]
            obj_index = [
                node_indexes[range][o] for o in triples[RDF_O].values]

            edge_index_matrices[(domain, edge_type, range)] = torch.tensor([sub_index, obj_index], dtype=torch.long)

        return edge_index_matrices



    def add_inverse_edges(self):
        for edge_type, edge_index in self.graph.edge_index_dict.items():
            inv_predicate = get_inverse(edge_type[1])
            if inv_predicate is None:
                continue

            inv_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            self.graph[(edge_type[2], inv_predicate, edge_type[0])].edge_index = inv_edge_index


    def create_node_labels(self, interaction_nodes, labels, interaction_indexes):
        """
        :param interaction_nodes: List[όλοι οι κόμβοι αυτού του τύπου (_:p)
        :param labels: Ο τύπος κάθε κόμβου interaction, κωδικοποιημένος με το Label Encoder
                Dict{key = '_:p{i}', value = Label_Encoder(interaction type του i)
        :param interaction_indexes: Dict{'_:p : A/A που αντιστοιχεί στον κόμβο interaction}
        :return: Tensor διάστασης : πλήθος κόμβων interactions με τα labels τους
        """
        n_nodes = len(interaction_nodes)
        labels_tensor = torch.empty(n_nodes, dtype=torch.long)
        for node in interaction_nodes:
            labels_tensor[interaction_indexes[node]] = torch.tensor(labels[node])
        return labels_tensor

# ----------------------------------------------------------------------------------------------------------------------

    def train_val_split(self, val_percentage=0.3):
        g_train = self.graph.clone()
        transform = RandomNodeSplit(num_val=val_percentage, num_test=0, num_splits=1)
        for store in g_train.node_stores:
            if transform.key is not None and not hasattr(store, transform.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[transform._split(store) for _ in range(transform.num_splits)])


            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        n_train_nodes = g_train[INTERACTION_CLASS].train_mask.count_nonzero().item()
        return g_train, n_train_nodes

    @staticmethod
    def get_dataloader(g_train, sample_size=200, itr=4, batch_size=324):

        g = torch.Generator()
        g.manual_seed(0)

        return HGTLoader(
            g_train,
            # Sample sample_size nodes per type and per iteration for itr iterations
            num_samples={key: [sample_size] * itr for key in g_train.node_types},
            # Use a batch size for sampling training nodes
            batch_size=batch_size,
            input_nodes=(INTERACTION_CLASS, g_train[INTERACTION_CLASS].train_mask),
            shuffle=True,
            drop_last=True,
            generator=g
        )

    @staticmethod
    def nodes_to_predict(graph, val_set = False):
        if val_set:
            indexes = graph[INTERACTION_CLASS].val_mask.nonzero(as_tuple=True)[0]
        else:
            indexes = graph[INTERACTION_CLASS].train_mask.nonzero(as_tuple=True)[0]
        labels  = graph[INTERACTION_CLASS].y[indexes]
        n_nodes = labels.size(0)
        return indexes, labels, n_nodes


    @staticmethod
    def graph_stats(graph, val_set=False):
        _, labels, _ = GraphDataset.nodes_to_predict(graph, val_set=val_set)
        print(labels.unique(return_counts=True))



