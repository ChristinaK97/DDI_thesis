import requests
import numpy as np
import pandas as pd
from thefuzz import process
from tqdm import tqdm
from os.path import exists

from other.file_paths import synonyms_db_file, drugbank_vocab_file
from source.preprocessing.parse_collection_xml_files import parse_name

C_NAME = 'c_name'
SYNON = 'synonym'


class Synonyms:

    def __init__(self, entities=None):
        """
        Δημιουργεί ή φορτώνει από το αρχείο synonyms.csv τη βάση συνωνώμων των ουσιών του
        dataset ddi.
        :param entities: Για να δημιουργηθεί πρώτη φορά η βάση (το csv δεν υπάρχει), δέχεται
               ένα set με τα ονόματα των οντοτήτων του ddi. Αλλιώς δεν χρειάζεται το όρισμα,
               απλά φορτώνει το αρχείο csv που έχει δημιουργηθεί σε προηγούμενη εκτέλεση.
        """
        try:
            self.synonyms = pd.read_csv(synonyms_db_file)
        except FileNotFoundError:
            self.make_db(entities)
            self.synonyms.to_csv(synonyms_db_file, sep=',', encoding='utf-8', index=False)  # 8


    def make_db(self, entities):
        """
        1. Διαβάζει το csv αρχείο drugbank vocabulary και κρατάει μόνο τις πληροφορίες:
           όνομα ουσίας (Common name) -> λίστα συνωνύμων (Synonyms)
        2. Για κάθε common name :
           3. Αν ανήκει στις οντότητες του dataset (ddi) :
              4. Κράτα κάθε συνώνυμο του, s, που ανήκει σε αυτές,
                 σε εγγραφή (common_name, s)
        """
        if exists(drugbank_vocab_file):

            df = pd.read_csv(drugbank_vocab_file)[['Common name', 'Synonyms']]  # 1

            self.synonyms = []
            for _, line in df.iterrows():  # 2
                try:
                    c_name = parse_name(line['Common name'])
                    if c_name not in entities:
                        continue
                    # 3
                    for synonym in line['Synonyms'].split('|'):  # 4
                        s = parse_name(synonym)
                        if s in entities:
                            self.synonyms.append((c_name, s))

                except AttributeError:
                    pass  # Η ουσία c_name δεν έχει συνώνυμα/Το πεδίο Synonyms είναι κενό
        else:
            self.synonyms.append(('-', '-'))
        self._continue_to_pubchem(entities)


    def _continue_to_pubchem(self, entities):
        """
        5. Φτιάχνει DF με εγγραφές (όνομα ουσίας c_name, συνώνυμο)
        6. Αντλεί επιπλέον συνώνυμα για τις οντότητες του ddi από το pubchem api
        7. Απορρίπτει τις διπλότυπες εγγραφές. Τα (x, y) (y,x) θεωρούνται ίδια και
           το ένα απορρίπτεται
        8. Γράφει το DF synonyms σε αρχείο synonyms.csv που αποτελεί βάση συνωνύμων
           των οντοτήτων του ddi set
        """
        self.synonyms = pd.DataFrame(self.synonyms, columns=[C_NAME, SYNON])  # 5

        self._add_synonyms_from_pubchem(entities)  # 6
        self.synonyms.sort_values([C_NAME, SYNON], inplace=True, ignore_index=True)

        self.synonyms = self.synonyms.loc[  # 7
            pd.DataFrame(np.sort(self.synonyms[[C_NAME, SYNON]], 1), index=self.synonyms.index)
                .drop_duplicates(keep='first').index
        ]


    def _add_synonyms_from_pubchem(self, entities):
        """
        Ανακτά από το pubchem api τα συνώνυμα των entities και τα προσθέτει στο DF των συνωνύμων.
        Δεν αποθηκεύει τις νέες εγγραφές στο αρχείο csv.
        :param entities: Ένα σύνολο ονομάτων οντοτήτων.
        :return: -
        """
        """
        1. Για κάθε οντότητα drug :
           2. Αν δεν υπάρχουν ήδη τα συνώνυμα της στην βάση (δεν υπάρχει είτε ως c_name
              ή synonym) :
              3. Ανακτά ένα σύνολο συνωνύμων και τα κρατά ως (drug, s)
        4. Προσθέτει τα νέα ζεύγη στο DF synonyms της βάσης.           
        """
        pairs = []
        for drug in tqdm(entities):  # 1
            if len(self.find_synonyms(drug)) == 0:  # 2
                try:
                    for s in req_synonyms(drug) :  # 3
                        pairs.append((drug, parse_name(s)))
                except: pass

        pairs = pd.DataFrame(pairs, columns=[C_NAME, SYNON])
        self.synonyms = self.synonyms.append(pairs, ignore_index=True)  # 4

# ----------------------------------------------------------------------------------------

    def find_collection_synonyms(self, unique_entities):
        """
        Δοθέντος ενός συνόλου διακριτών οντοτήτων, εντοπίζει τα συνώνυμα τους στη βάση
        και επιστρέφει τα ζεύγη των συνώνυμων.
        :param unique_entities: Set με τα ονόματα των οντοτήτων. Οι οντότητες πρέπει να
        έχουν αποθηκευτεί σε προηγούμενη φάση στη βάση.
        :return: Set(Tuple(drug1, drug2)) όπου drug1 και drug2 είναι συνώνυμα
        """
        synonyms_found = set()

        for _, pair in self.synonyms.iterrows():
            c_name = pair[C_NAME]
            syn = pair[SYNON]
            if c_name != syn and c_name in unique_entities and syn in unique_entities:
                synonyms_found.add((c_name, syn))
        return synonyms_found




    def check_if_synonyms(self, w1, w2):
        """
        Ελέγχει με fuzzy τρόπο αν ουσίες με όνοματα w1 και w2 είναι εναλλακτική
        ονομασία η μία της άλλης.
        :return: True αν είναι συνώνυμα
        """
        # Έλεγχει αν το w2 έχει ομοιότητα τουλ 97% με κάποιο συνώνυμο του w1,
        # ή το αντίστροφο
        if fuzzycheck_if_in_synonyms(w2, self.find_synonyms(w1)) \
        or fuzzycheck_if_in_synonyms(w1, self.find_synonyms(w2)):
            return True
        else:
            return False


    def find_synonyms(self, drug_name):
        """
        Βρίσκει τα συνώνυμα μιας ουσίας από τη βάση που έχει δημιουργηθεί. Δεν βρίσκει
        συνώνυμα για ουσίες που δεν έχει ήδη αποθηκεύσει.
        :param drug_name: Το όνομα της ουσίας που θα αναζητήσει τα συνώνυμα, ανεξάρτητα
                          αν είναι c_name λη synonym
        :return: Ένα σετ με τα συνώνυμα, ή αν η ουσία δεν είναι στη βάση κενό σετ
        """
        """
        1. Ψάχνει αν το drug_name είναι c_name, δλδ αν συμμετέχει σε ζεύγη (drug_name, ?s)
        2. Αν δεν το βρει ως c_name, θα προσπαθήσει να το βρει ως synonym : (?c_name, drug_name)
           3. Αν το βρεί ως synonym, θα ανακτήσει το c_name στο οποίο αντιστοιχεί
           4. Θα βρει τα συνώνυμα του c_name (c_name, ?s), που αποτελούν και συνώνυμα του drug_name
           5. Θα αφαιρέσει το ίδιο από το συνολο των συνωνύμων
           6. Θα προσθέσει το c_name ως συνώνυμο του drug_name
           7. Αλλιώς αν δεν βρέθηκε (?c_name, drug_name) => το drug_name δεν υπάρχει στη βάση
        """
        syn = self._find_synonyms_by_cname(drug_name)  # 1

        if len(syn) == 0:  # 2
            try:
                c_name = self.synonyms[self.synonyms[SYNON] == drug_name][C_NAME].values[0]  # 3
                syn = self._find_synonyms_by_cname(c_name)  # 4
                syn.remove(drug_name)  # 5
                syn.add(c_name)  # 6
            except IndexError:   # 7
                syn = set()
        return syn


    def _find_synonyms_by_cname(self, drug_name):
        """
        Επιστρέφει σύνολο με τις εναλλακτικές ονομασίες ?s του drug_name
        Συμμετέχουν στα ζεύγη της μορφής (drug_name, ?s)
        :param drug_name: Η ουσία που ψάχνει ως c_name για να γυρίσει τα συνώνυμα της
        :return: Ένα σύνολο με τα συνώνυμα του c_name
        """
        return set(self.synonyms[self.synonyms[C_NAME] == drug_name][SYNON])



# --------------------------------------------------------------------------------------


def req_synonyms(drug_name):
    """
    Ανακτά από το pubchem api και επιστρέφει ένα σύνολο εναλλακτικών ονομάτων για
    την ουσία drug_name.
    :param drug_name: Το όνομα μιας ουσίας
    :return: Σετ με τα συνώνυμα, ή αθλίως αν δε βρήκε ή προέκυψε κάποιο σφάλμα
             επιστρέφει κενό σύνολο
    """
    query = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + drug_name + '/synonyms/JSON'
    try:
        response = requests.get(query)
    except Exception:
        return set()

    if response.status_code < 400:
        return response.json()['InformationList']['Information'][0]['Synonym']
    return set()



def fuzzycheck_if_in_synonyms(drug_name, synonyms):
    """
    Ελέγχει με fuzzy τρόπο (97% κάτω όριο ομοιότητας αν η ουσία με όνομα drug_name
    ανήκει σε ένα σύνολο των συνωνύμων synonyms
    :return: True αν ανήκει
    """
    try:
        # Εύρεση του κοντινότερου αντικειμένου στο drug_name από το σύνολο synonyms
        closest_match = process.extractOne(drug_name, synonyms)
        # Ελέγχει αν η ομοιότητα τους είναι τουλ 97%
        return closest_match[1] >= 97
    except TypeError:
        return False






