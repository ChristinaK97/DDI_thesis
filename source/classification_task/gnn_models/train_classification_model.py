import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch_geometric.utils import dropout_adj
from torchmetrics import ConfusionMatrix

from other.CONSTANTS import INTERACTION_CLASSES, INTERACTION_CLASS
from other.utils import set_seed, device
from source.classification_task.dataset_preparation.graph_dataset import GraphDataset
from source.classification_task.gnn_models.analyse_misclassified import run_miscls
from source.classification_task.gnn_models.load_classification_model import ClassificationModel, \
    files_found
from config import *

TRAINING = 1
INFERENCE = 2

pd.set_option('display.width', 500)
pd.set_option("display.max_rows", None, "display.max_columns", None)
# ================================================================================================


class TrainClassificationModel:

    def __init__(self, mode=TRAINING):

        set_seed()
        self.device = device()
        self.whole_set = True if VAL_PREC == 0 else False

        if not files_found():
            mode = TRAINING

        if mode == TRAINING:

            self.dataset, self.g_train, self.n_interaction_nodes, \
                self.loader, self.g_test = self.prepair_dataset()

            self.save_model_config()
            self.model = ClassificationModel()

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            self.train_model()
            torch.save(self.model.state_dict(), MODEL_FILE)

            self.evaluate_test_set()
        else:
            self.inference()


    def inference(self):
        """
        Φορτώνει το εκπαιδευμένο μοντέλο και το εφαρμόζει στο σύνολο ελέγχου
        """
        self.model = ClassificationModel(load_model=True)
        self.evaluate_test_set()



    def prepair_dataset(self):
        """
        Δημιουργεί ή φορτώνει αν υπάρχουν αποθηκευμένα τα σύνολα εκπαίδευσης και ελέγχου
        :return:
            dataset: Αντικείμενο Graph_Dataset για το σύνολο εκπαίδευσης
            g_train: HeteroData ο γράφος του συνόλου εκπαίδευσης
            n_train_nodes: Το πλήθος των κόμβων Interaction που χρησιμοποιούνται για εκπαίδευση
            loader: HGTLoader Dataloader για minibatch training
            g_test: HeteroData ο γράφος του συνόλου ελέγχου
        """

        dataset = GraphDataset(train=True)
        g_train, n_train_nodes = (dataset.graph, dataset.n_interaction_nodes) if self.whole_set else \
                                  dataset.train_val_split(val_percentage=VAL_PREC)
        g_train.to(self.device)
        GraphDataset.graph_stats(g_train)
        loader = dataset.get_dataloader(g_train)

        g_test = GraphDataset(train=False).graph
        g_test.to(self.device)

        return dataset, g_train, n_train_nodes, loader, g_test



    def save_model_config(self):
        """
        Δημιουργεί το αρχείο classification_model_config.json με τις παράμετρους
        του μοντέλου όπως έχουν οριστεί στο config.py
        """
        model_config = {
            'in_channels': self.dataset.emb_dim,
            'node_types': self.dataset.graph.metadata()[0],
            'edge_types': self.dataset.graph.metadata()[1],

            'preproc': MLP_PREPROCESSING_DIM,
            'postproc': MLP_POSTPROCESSING_DIM,
            'n_conv_layers': 1,
            'hidden_channels': HIDDEN_CHANNELS,
            'conv_type': GNN_TYPE,

            'act_func': ACTIVATION_FUNC
        }
        with open(MODEL_CONFIG_FILE, "w") as fp:
            json.dump(model_config, fp)

# ================================================================================================

    def train_model(self):
        """
        Εκτελεί την εκπαίδευση του μοντέλου και τυπώνει τα αποτελέσματα για κάθε σετ
        ανά 10 εποχές.
        """
        with torch.no_grad():  # lazy init
            nodes, _, _ = self.get_labels(self.g_train)
            self.model(data=self.g_train, nodes_to_predict=nodes)

        for epoch in range(EPOCHS):
            # mini-batch
            running_loss = sum([self.train(batch) for batch in self.loader]) / self.n_interaction_nodes

            if epoch % 10 == 0:
                results = {}
                results['TRAIN'] = self.evaluate(self.g_train)
                if not self.whole_set:
                    results['VAL'] = self.evaluate(self.g_train, val_set=True)
                results['TEST'] = self.evaluate(self.g_test, show_confmat=(epoch == EPOCHS), get_per_class=True)
                self.print_results(epoch, running_loss, results)


# ================================================================================================

    def get_labels(self, data, val_set=False):
        """
        :param data: Αντικείμενο HeteroData. Μπορεί να είναι ένα batch του loader
                     ή ολόκληρο σύνολο
        :param val_set: Αν πρόκειται για το σύνολο επικύρωσης True (default=False)
        :return: nodes_to_predict (tensor) : Τα indexes των κόμβων Interaction που θα προβλεφθεί
                 η κλάση τους από το μοντέλο
                 labels (tensor) : Η πραγματική κλάση κάθε τέτοιου κόμβου
                 n_nodes (int) : Το πλήθος τους
        """
        nodes_to_predict, labels, n_nodes = GraphDataset.nodes_to_predict(data, val_set=val_set)

        return nodes_to_predict, labels, n_nodes

# ================================================================================================

    def train(self, batch):
        """
        Εκτελεί μία εποχή εκπαίδευσης για ένα batch
        :param batch: HeteroData Ένα κομμάτι του γράφου από τον loader
        :return: Το running loss της εποχής στο συγκεκριμένο batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        # batch = self.drop_edges(batch)

        nodes_to_predict, batch_labels, n_nodes = self.get_labels(batch)
        y_hat = self.model(data=batch, nodes_to_predict=nodes_to_predict)

        loss = self.criterion(y_hat, batch_labels)
        loss.backward()
        self.optimizer.step()

        return loss.item() * n_nodes


    def drop_edges(self, batch):
        """
        Αφαιρεί κάποιες ακμές από το γράφο batch
        :param batch: HeteroData Ένα κομμάτι του γράφου από τον loader
        :return: Το ίδιο αντικείμενο με p % τυχαία απορριφθέντες ακμές
                  για κάθε τύπο ακμής
        """
        for edge_type in batch.metadata()[1]:
            edge_index = batch[edge_type].edge_index
            edge_index, _ = dropout_adj(edge_index, p=0.4, training=True)
            batch[edge_type].edge_index = edge_index
        return batch

# ============================================================================================================

    def evaluate_test_set(self):
        """
        Εφαρμόζει το εκπαιδευμένο μοντέλο για το σύνολο ελέγχου κατά
        την λειτουργία INFERENCE.
        """
        self.dataset = GraphDataset(train=False)
        g_test = self.dataset.graph
        g_test.to(self.device)
        results, y_hat, labels = self.evaluate(g_test, show_confmat=True, get_per_class=True, error_an=True)
        self.print_results('INFERENCE', 0., {'TEST': results})
        self.incorrect_pairs(self.dataset.node_indexes[INTERACTION_CLASS], y_hat, labels)


    def incorrect_pairs(self, pairs, y_hat, labels):
        """
        Εντοπίζει τα λάθη που έγιναν από το μοντέλο. Αν στο config.py
        οριστεί PRINT_MISCLS=True τότε θα παράγει λεπτομερή αναφορά
        των σφαλμάτων.
        :param pairs: Dict{interaction_node_id : index στο GraphDataset}
        :param y_hat: Οι προβλέψεις του μοντέλου
        :param labels: Οι πραγματική κλάση κάθε κόμβου interaction
        """
        pairs = {index: pair for pair, index in pairs.items()}
        errors = []

        for i, (cl, pred) in enumerate(zip(labels, y_hat)):
            if cl == pred: continue
            errors.append({INTERACTION_CLASS: pairs[i], 'y': cl.item(), 'y_hat': pred.item()})

        errors = pd.DataFrame(errors)
        for col in ['y', 'y_hat']:
            errors[col] = self.dataset.label_enc.inverse_transform(errors[col])
        print(' test set size =', labels.size(0))
        print('misclassified  =', errors.shape[0])
        run_miscls(errors, print_results=PRINT_MISCLS)




    def evaluate(self, data, show_confmat=False, get_per_class=False, error_an=False, val_set=False):
        """
        :param data: HeteroData ένα σύνολο γράφου
        :param show_confmat: True αν θα εμφανίσει πίνακα σύγχυσης (default False)
        :param get_per_class: True αν θα δώσει αποτελέσματα για κάθε κλάση ξεχωριστά (default False)
        :param error_an: True αν θα επιστρέψει επιπλέον τις προβλέψεις του μοντέλου και τα labels (default False)
        :param val_set: True αν επιθυμείται εκτίμηση στο validation set (default False)
        :return: results : 2D Dict Για κάθε μετρική επιστρέφει το αποτέλεσμα
                Αν error_an == True επιστρέφει επιπλέον:
                 y_hat : Οι προβλέψεις του μοντέλου
                 eval_labels : Οι πραγματικές κλάσεις
        """
        # forward pass
        with torch.no_grad():
            self.model.eval()
            nodes_to_predict, eval_labels, n_nodes = self.get_labels(data, val_set=val_set)
            y_hat, _ = self.model(data=data, nodes_to_predict=nodes_to_predict)

        y_hat_detection = self.turn_to_binary(y_hat)
        labels_detection = self.turn_to_binary(eval_labels)

        # ------------------------------------------------------------
        # METRICS
        # ------------------------------------------------------------
        precision_metric = torchmetrics.functional.classification.precision
        recall_metric    = torchmetrics.functional.classification.recall
        f1_metric        = torchmetrics.functional.classification.f_beta.f1_score

        if show_confmat:
            self.conf_matrix(y_hat, eval_labels)

        results = {}

        for metric in [f1_metric, precision_metric, recall_metric]:

            metric_name = metric.__name__
            results[metric_name] = {}
            results[metric_name]['micro'] = metric(y_hat, eval_labels).item()
            results[metric_name]['micro+'] = metric(y_hat, eval_labels, ignore_index=self.dataset.negative_label).item()
            results[metric_name]['binary'] = metric(y_hat_detection, labels_detection).item()

            if get_per_class:
                metric_per_class = metric(y_hat, eval_labels, average=None,
                                          num_classes=len(INTERACTION_CLASSES))

                for class_, value in enumerate(metric_per_class):
                    results[metric_name][self.dataset.label_enc.inverse_transform([class_])[0]] = value.item()

        return results if not error_an else (results, y_hat, eval_labels)


    def turn_to_binary(self, labels):
        """
        Μετατρέπει τις τιμές του labels σε δυαδικές.
        Για κάθε l στο labels αν είναι η αρνητική κλάση True, για τις άλλες False
        """
        return torch.tensor([l == self.dataset.negative_label for l in labels], dtype=torch.bool)


    def conf_matrix(self, y_hat, labels):
        """
        Δημιουργεί και εμφανίζει τον πίνακα σύγχυσης.
        """
        confmat = ConfusionMatrix(num_classes=len(self.dataset.label_enc.classes_))
        confmat.to(self.device)
        cm = confmat(y_hat, labels).cpu().detach().numpy()

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.dataset.label_enc.classes_)
        disp.plot()
        plt.show()

# =========================================================================================================
    def print_results(self, epoch, running_loss, results):

        print(f'> Epoch {epoch}\n\tLoss = {"{:.5f}".format(running_loss)}')
        for set_, values in results.items():
            values = pd.DataFrame(values).T
            print(f'{set_}\n{values}')

        print('_' * 100)
# =========================================================================================================
