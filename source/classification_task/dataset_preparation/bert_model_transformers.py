from os.path import exists

import torch
import torch.nn as nn
from config import BERT_MODEL_NAME
from transformers import BertModel
from tqdm import tqdm

from other.file_paths import graph_dataset
from source.classification_task.dataset_preparation.bert_dataset import \
    Bert_Dataset, SPECIAL_TOKENS, SENTENCE_EMB, TOKEN_EMB
from other.utils import set_seed

# ------------------------------------------------------------------------------------------

AVG_POOL = 'avg'
MAX_POOL = 'max'
SELECTED_POOL = AVG_POOL


class BioBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

    def forward(self, input_data, embeddings_mode):
        """
        :param input_data: Εχει τα πεδία:
               - input_ids: Tensor με τα ids των bert tokens της πρότασης
               - attention_mask: Tensor μάσκα, ίδιου μήκους
               - units_identifiers:

                Για παραγωγή token embeddings :
                    Dict{key = named_entity Token -> όνομα του κόμβου στο γράφο,
                         value = List με τα indexes που αφορούν την οντότητα}
                Για παραγωγή sentence embeddings :
                    str : sentence identifier (COLLECTION.dx.sy)

        :return:
            Για παραγωγή token embeddings :

                Dict{key = named_entity Token, value = Tensor : Το embedding της οντότητας σε αυτή την πρόταση]

                                                        1o κομμάτι   2ο κομμ
                                  {units_identifiers : [3, 4, 5,     10, 11, 12], ...}
                token_embeddings  {units_identifiers : POOL(s_embeddings[i] i=3,4,5,10,11,12)}

            Για sentence embeddings :
                Dict{key = sentences_identifies από το γράφο (COLLECTION.dx.sy),
                     value = Tensor το embedding της πρότασης}
        """
        with torch.no_grad():
            units_identifiers, input_ids, attention_mask = input_data
            encoded_layer = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            s_embeddings = encoded_layer.last_hidden_state[0]

        return self.named_entities_embeddings(s_embeddings, units_identifiers) if embeddings_mode == TOKEN_EMB \
            else self.sentence_words_embeddings(s_embeddings, units_identifiers)


    def named_entities_embeddings(self, s_embeddings, s_ne_indexes):
        """
        Βρίσκει τα embeddings των named entities της πρότασης
        :param s_embeddings: Tα emdeddings των οντοτήτων της πρότασης για την οποία εφαρμόστηκε το μοντέλο
        """
        entities_embeddings = {}

        for entity in s_ne_indexes:
            if entity in SPECIAL_TOKENS: continue

            s = self.pooling([s_embeddings[i] for i in s_ne_indexes[entity]], pool_method=SELECTED_POOL)
            entities_embeddings[entity] = s
        return entities_embeddings


    def sentence_words_embeddings(self, s_embeddings, sentence_id):

        #p = self.pooling([emb for emb in s_embeddings[1:-1]], pool_method=SELECTED_POOL)
        p = s_embeddings[0]  # cls
        return {sentence_id: p}



    def pooling(self, embeddings, pool_method):
        """
        Εφαρμογή pooling σε μια λίστα από embeddings
        :param embeddings: List[Tensor] embeddings για μία οντότητα
        :param pool_method: 'avg' ή 'max' για τον συνδυασμό τους
        :return: Tensor το pooled embedding
        """
        embeddings = torch.stack(embeddings)

        if pool_method == AVG_POOL:
            return torch.sum(embeddings, dim=0, dtype=torch.float32) \
                   / len(embeddings)

        elif pool_method == MAX_POOL:
            return torch.max(embeddings, 0)[0]

# ======================================================================================================================

class WordEmbeddings:

    def __init__(self, train, sentences_init=None):
        """
        :param train: Αν αναφέρεται στο σύνολο εκπαίδευσης True. False για το Test set
        :param sentences_init: Αποτέλεσμα του query q_collect_sentences() στο neo4j (αρχείο query_neo4j)
                Λίστα που περιέχει όλες τις προτάσεις της βάσης, μαζί με τις named entities της πρότασης και τα όρια
                start και end κάθε ne.
                List[List[sentence_text,
                          List[Dict{'Drug_Class':named_entity,
                                    'Token': unique identifier του κόμβου που αναφέρεται στην εμφάνιση
                                             του συγκεκριμένου φαρμάκου στη συγκεκριμένη πρόταση,
                                    'start':char_Offset_start,
                                     'end':char_Offset_ent}],
                          Sentence identifier (COLLECTION.d{x}.s{y})]
                Αν τα αρχεία υπάρχουν, δεν χρειάζεται το όρισμα
        """
        self.sentence_emb_file = WordEmbeddings.embeddings_file_path(train, SENTENCE_EMB)
        self.token_emb_file    = WordEmbeddings.embeddings_file_path(train, TOKEN_EMB)
        self.files_exist       = exists(self.sentence_emb_file) and exists(self.token_emb_file)
        if not self.files_exist:
            if sentences_init is None:
                raise Exception('Files not found and sentence_init arg eq to None')

            self.sentences_init = sentences_init
            set_seed()
            self.model = BioBert().eval()



    @staticmethod
    def embeddings_file_path(train, embeddings_mode):
        return graph_dataset(train) + '/raw/' + \
            ('sentence' if embeddings_mode == SENTENCE_EMB else 'token')\
            + '_embeddings.pt'



    def word_embeddings(self):
        """
        Βρίσκει τα word embeddings των sentences και των tokens τους στη συλλογή.
        Αν τα έχει υπολογίσει ήδη τα φορτώνει από το αρχείο
        αλλίως εφαρμόζει το μοντέλο για να τα βρει.
        :return: Dict{key = arg1 : sentence/ arg2 : token identifier -> όπως εμφανίζεται στον κόμβο του γράφου,
                      value = word embedding την οντότητας}
        """
        embeddings = {}
        for (file, embeddings_mode) in [(self.sentence_emb_file, SENTENCE_EMB),(self.token_emb_file, TOKEN_EMB)]:

            embeddings[embeddings_mode] = \
                        torch.load(file) if self.files_exist \
                        else self._run_bert(file, embeddings_mode)

        return embeddings[SENTENCE_EMB], embeddings[TOKEN_EMB]


    def _run_bert(self, embeddings_file, embeddings_mode):
        print('BERT Inference...')
        embeddings = {}
        data = Bert_Dataset(self.sentences_init, embeddings_mode)

        for i in tqdm(range(len(data))):
            input_data = data.get(i)
            bert_output = self.model(input_data, embeddings_mode)

            for unit_identifier, embedding in bert_output.items():
                embeddings[unit_identifier] = embedding

        torch.save(embeddings, embeddings_file)

        return embeddings


