import pickle

import pandas as pd
import torch
from config import PROJECT_PATH
from pytorch_pretrained_bert import BertTokenizer

from other.CONSTANTS import *

SENTENCE_EMB = 1
TOKEN_EMB = 2
SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]']

class Bert_Dataset():

    def __init__(self, sentences_init, bert_path, embeddings_mode, run_for_bert=True):
        """
        Τα δεδομένα (sentences) που θα χρησιμοποιήσει το biobert για να βγάλει τα embeddings των named entities.

        :param sentences_init: Αποτέλεσμα του query q_collect_sentences() στο neo4j (αρχείο query_neo4j)
            Λίστα που περιέχει όλες τις προτάσεις της βάσης, μαζί με τις named entities της πρότασης και τα όρια
            start και end κάθε ne. Τρίτο όρισμα, ο identifier της πρότασης στον γράφο.
                 s    s[0]           s[1]  i                        i[start]                                            s[2]
            List[List[sentence_text, List[Dict{'drug':named_entity, 'start':char_Offset_start, 'end':char_Offset_end}], sent_id]
        :param bert_path: Path του φακέλου όπου είναι αποθηκευμένο το μοντέλο biobert
        """
        """
        1. Το text των sentences. [1:-1] : αφαιρεί τα περιττά επιπλέον "" που επέστρεψε το neo4j
        2. sentences : Το κείμενο κάθε πρότασης, 
        3. sentences_identifies από το γράφο (COLLECTION.dx.sy)
        4. Παράλληλη λίστα με τα sentences. Περιέχει τα στοιχεία των ne της κάθε πρότασης. Ταξινόμηση κατά
           το start index, ώστε τα ne να αποθηκευτούν με την σειρά που εμφανίζονται μέσα στην πρόταση
        5. Τα ids που αντιστοιχούν στα bert tokens σύμφωνα με το biobert vocab
        """
        self.embeddings_mode = embeddings_mode

        self.tokenizer = BertTokenizer(vocab_file=bert_path + 'vocab.txt', do_lower_case=False)  # 1
        self.sentences = pd.Series([s[0][1:-1] for s in sentences_init])                         # 2
        self.sentences_identifiers = pd.Series([s[2] for s in sentences_init])                   # 3
        self.named_entities = [sorted(s[1], key=lambda i: i[START]) for s in sentences_init]     # 4

        if self.embeddings_mode == TOKEN_EMB:
            self.token_to_drug = self.find_token_to_drug()

        if run_for_bert:
            self.ids, self.attention_masks = self.prepare_data()  # 5
        else:
            self.save_sentences()
        # MAX_LEN = max([sentence_ids.size(1) for sentence_ids in self.ids])


    def get(self, item):
        """
        :param item: Index της πρότασης
        :return: Για mode: token_embeddings τη λίστα (mask) των named entity tokens,
                           sentence_embeddings Tuple(sentence identifier, πλήθος των bert token ids της πρότασης)
                 τα tensor ids και attention mask που αντιστοιχούν στην bert tokens αναπαράσταση της πρότασης
        """

        return (self.named_entities[item] if self.embeddings_mode == TOKEN_EMB
          else self.sentences_identifiers[item]),  \
          self.ids[item], self.attention_masks[item]


    def __len__(self):
        """
        :return: Το πλήθος των προτάσεων της βάσης
        """
        return len(self.ids)


    def prepare_data(self):
        """
        :return: Λίστα των ids (tensors) των προτάσεων που αντιστοιχούν στα bert tokens σύμφωνα με το biobert vocab
                 Λίστα των attention masks (tensors) των bert tokens για κάθε πρόταση
        """
        """
        1. Διάσπαση του συνεχούς κειμένου της πρότασης σε tokens (χωρισμένα με ' '), με ειδική διαχείριση των ne
        2. Αντικατάσταση των ονομάτων κάθε drug με drug{A/A} για παραγωγή token embeddings, drug0 για sentence embeddings
        3. Επιπλέον διάσπαση σε bert tokens σύμφωνα με τον tokenizer
        4. Μετατροπή των bert tokens σε ids σύμφωνα με το biobert vocab
        5. Εντοπισμός των indexes στον πίνακα των embeddings της πρότασης, όπου αφορά την κάθε ne 
        """
        self.apply_transformation(self.tokenize_sentence)   # 1
        self.apply_transformation(self.mask_tokens)   # 2
        self.apply_transformation(self.bert_tokenizer) # 3

        ids = [self.tokenizer.convert_tokens_to_ids(s) for s in self.sentences]  # 4
        attention_masks = self.make_attention_masks()  # 5

        ids = self.to_tensors(ids)
        attention_masks = self.to_tensors(attention_masks)

        self.named_entities_to_indexes()  # 5

        return ids, attention_masks


    def apply_transformation(self, trfm):
        """
        Εφαρμογή μίας συνάρτησης μετασχηματισμού trfm στις λίστες sentences και named_entities
        """
        self.sentences, self.named_entities = map(list, zip(*[
            trfm(s, t) for s, t in
            zip(self.sentences, self.named_entities)]))


    @staticmethod
    def get_tensor(el):
        """
        :param el: Μια λίστα
        :return: Η λίστα σαν tensor
        """
        tensor = torch.tensor(el)
        return tensor.view(-1, tensor.size()[-1])


    def to_tensors(self, list_):
        """
        Μετατρέπει τα στοιχεία (λίστες) μιας λίστας σε tensors
        και επιστρέφει αυτά τα tensors σε μία λίστα
        """
        return [self.get_tensor(el) for el in list_]


# ---------------------------------------------------------------------------------------------------------
    # 1
    def tokenize_sentence(self, s, marked_tokens):
        """
        Σπάει την δοθείσα πρόταση σε tokens, ενώ διαχειρίζεται ειδικά τα named entities.
        Επιπλέον θα προσθέσει τα ειδικά bert tokens CLS, SEP στην αρχή και το τέλος αντίστοιχα.
        :param s: Μια πρόταση (str)
        :param marked_tokens: Λίστα με τα named entities της πρότασης. Κάθε στοιχείο της λίστας
               είναι της μορφής Dict{'Drug_Clases':named_entity, 'Token':unique identifier του κόμβου Token
                                        που αντιστοιχεί στο συγκεκριμένο φάρμακο μέσα στην συγκεκριμένη πρόταση
                                     'start':char_Offset_start, 'end':char_Offset_ent}
        :return: sentence_words: Λίστα με τα tokens της πρότασης.
                 named_entities: Λίστα μορφής μάσκας, παράλληλη με την sentence_words.
                 Αν το token sentence_words[i] είναι named entity από τα marked tokens,
                 τότε το named_entities[i] θα έχει το όνομα του αντίστοιχου κόμβου Token του KG.

                 Το sentence_words περιέχει τα ne όπως αυτά εμφανίζονται μέσα στο κείμενο,
                 ενώ το named_entities όπως αναφέρονται στον κόμβο Token της οντότητας στο KG.
        """
        """
        Για κάθε marked token drug με start και end indexes μέσα στην πρόταση,
        - θα πάρει το substring πριν το start και θα το διασπάσει σε tokens με del=' '
          κρατώντας τα στη λίστα sentence_words
        - θα προσθέσει στα sentence_words το marked token (drug name) ακριβώς όπως εμφανίζεται
          στο κείμενο, χωρίς να το διασπάσει 
        - για κάθε token που δεν είναι named entity θα προσθέσει ένα '' στη λίστα named_entities
          και το drug name στην θέση του token του named entity
        """
        def tokenize_substr(st, en):
            nonlocal sentence_words, named_entities
            substr = s[st: en].split(' ')
            substr = list(filter(bool, substr))
            sentence_words += substr

            named_entities += ['' for _ in range(len(substr))]

        sentence_words = ['[CLS]']
        named_entities = ['[CLS]']
        start = 0

        for t in marked_tokens:
            end = t[START]
            start = start - 1 if start > 0 else start

            tokenize_substr(start, end)
            sentence_words.append(s[t[START] : t[END] + 1])
            named_entities.append(t[TOKEN_CLASS])

            start = t[END] + 2

        # αν υπολείπεται κείμενο μετά το τελευταίο marked token (δεν ήταν το τελευταίο token της
        # πρότασης => σπάει σε tokens και την υπόλοιπη συβ/ρά μέχρι το τέλος
        if start - 1 < len(s):
            tokenize_substr(start - 1, len(s))

        sentence_words.append('[SEP]')
        named_entities.append('[SEP]')

        return sentence_words, named_entities

# ---------------------------------------------------------------------------------------------------------

    # 2
    def find_token_to_drug(self):
        drug_tokens = {}
        for marked_tokens in self.named_entities:
            for t in marked_tokens:
                drug_tokens[t[TOKEN_CLASS]] = t[DRUG_CLASS]
        return drug_tokens


    def mask_tokens(self, sentence_words, named_entities):
        """
        Στη λίστα με τις λέξεις της πρότασης, αντικαθιστά τα named entities (φάρμακα) με 'DRUGx'
        όπου x = 0, για παραγωγή sentence embeddings (δεν ξεχωρίζουν μεταξύ τους τα φάρμακα)
               = A/A για κάθε φάρμακο μέσα στην πρότασης (DRUG1, ... ,DRUGN)
        :param sentence_words: Λίστα με τις λέξεις μίας πρότασης (μετά το sentence tokenizer, πριν το bert tokenizer)
        :param named_entities: Παράλληλη λίστα με μαρκαριμένες της θέσεις όπου εμφανίζεται ένα ne.
        :return: sentence_words, named_entities->αμετάβλητο
        """
        """
        1. indexes όπου εμφανίζεται το κάθε named entity μέσα στη λίστα λέξεων
        2. Για κάθε Token, List[indexes] :
           3. Αν δεν είναι ένα από τα ειδικά bert tokens :
              4. Κάνε την αντικατάσταση του με DRUGx (Σε κάθε θέση που το αποτελεί)
        """
        tokens_indexes = self.parse_sentence_ne(named_entities)        # 1
        if self.embeddings_mode == TOKEN_EMB:
            sentence_drugs = {self.token_to_drug.get(token, None) for token in tokens_indexes}
            sentence_drugs = {drug : 'drug' + str(i) for i, drug in enumerate(sentence_drugs) if drug is not None}

        for token, indexes in tokens_indexes.items():  # 2
            if token in SPECIAL_TOKENS : continue  # 3

            for index in indexes:  # 4
                sentence_words[index] = 'drug0' if self.embeddings_mode == SENTENCE_EMB else \
                        sentence_drugs[self.token_to_drug[token]]

        return sentence_words, named_entities

# ---------------------------------------------------------------------------------------------------------

    # 3
    def bert_tokenizer(self, sentence_words, named_entities):
        """
        Εφαρμογή του bert tokenizer στα tokens μιας πρότασης
        Πηγή: https://github.com/perkdrew/advanced-nlp/blob/master/BioBERT/ner/biobert_ner.ipynb
        :param sentence_words: Λίστα με τα tokens μιας πρότασης
        :param named_entities: Παράλληλη λίστα με μαρκαριμένες της θέσεις όπου εμφανίζεται ένα ne.
        :return: bert_tokens: Λίστα με τα tokens της πρότασης αφού έχει γίνει αντιστοίχηση/διάσπαση
                 με τον bert tokenizer σύμφωνα με το biobert vocab
                 named_entities_ext: Παράλληλη λίστα με την bert_tokens

        [...,token1, bacteria,     token2,...] ~ [...,'ba',           '##cter',      '##ia',       ...]
        [...,  ''  , bacteria_node,  ''  ,...] ~ [..., bacteria_node, bacteria_node, bacteria_node,...]
        """
        bert_tokens = []
        named_entities_ext = []
        for word, ne in zip(sentence_words, named_entities):
            word_tokens = self.tokenizer.tokenize(word)
            n_tokens = len(word_tokens)

            bert_tokens.extend(word_tokens)
            named_entities_ext.extend([ne] * n_tokens)
        return bert_tokens, named_entities_ext

# ---------------------------------------------------------------------------------------------------------

    def parse_sentence_ne(self, sentence_named_ent):
        """
        Μετατρέπει τη λίστα μάσκα named_entities μιας πρότασης σε λίστα από indexes
        List με τα indexes που αφορούν την οντότητα Token

                    1ο κομμάτι                        2ο κομμάτι
        [..., <3> ne, <4> ne, <5> ne,..., <10> ne, <11> ne, <12> ne, ...]   =>

        {ne : [3, 4, 5, 10, 11, 12], ...}
        """
        ne_indexes = {k: [] for k in set(sentence_named_ent)}
        ne_indexes.pop('', None)

        for i, token in enumerate(sentence_named_ent):
            if token != '':
                ne_indexes[token].append(i)

        return ne_indexes



    def named_entities_to_indexes(self):
        """
        Μετατρέπει τη λίστα μάσκα named_entities κάθε πρότασης σε λίστα από indexes
        Dict{key=named_entity -> όνομα του κόμβου Token στο γράφο, value = List με τα indexes που αφορούν την οντότητα}
        """
        self.named_entities = [
            self.parse_sentence_ne(s) for s in self.named_entities
        ]


    def make_attention_masks(self):
        """
        Παράγει το attetion mask για τα bert token ids κάθε πρότασης.
        Για παραγωγή sentence embeddings κάθε id (bert token) θα ληφθεί υπόψη
            - Παντού 1, [1] * πλήθος των bert token ids
        Για token embeddings, μόνο τα φάρμακα θα ληφθούν υπόψη
            - 1 στην θέση i, αν το ids[i] αντιστοιχεί σε bert token φαρμάκου.
        """
        return [[1]*len(sentence) for sentence in self.named_entities]

# ---------------------------------------------------------------------------------------------------------

    def save_sentences(self):
        self.apply_transformation(self.tokenize_sentence)  # 1
        self.apply_transformation(self.mask_tokens)  # 2
        dict = {}
        for i in range(len(self.sentences)):
            dict[self.sentences_identifiers[i]] = self.sentences[i]

        a_file = open(f"{PROJECT_PATH}data/models/sentences.pkl", "wb")
        pickle.dump(dict, a_file)
        a_file.close()
