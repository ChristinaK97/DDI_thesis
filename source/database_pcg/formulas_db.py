import sqlite3
import requests
from os import path
from tqdm import tqdm

from other.file_paths import formulas_db_file

FORMULAS_TABLE = 'Formulas'
ENTITY = 'entity'
FORMULA = 'formula'

class FormulasDB:
    # Dict {όνομα ουσίας:μοριακός τύπος}, όπου θα διατηρηθούν όσα ανάκτησε κατά τη διαδικασία
    in_memory_formulas = {}

    def __init__(self, entities=None):
        """
        Δημιουργεί σύνδεση με τη βάση των μοριακών τύπων. Αν δεν υπήρχε την δημιουργεί
        σύμφωνα με το όρισμα entities.
        :param entities: Σετ μοναδικών οντοτήτων της συλλογής που θα εισαχθούν στη νέα
            βάση. Αν η βάση ήδη υπάρχει, δεν εισάγονται οι entities.
        """
        create = not path.exists(formulas_db_file)

        self.conn = sqlite3.connect(formulas_db_file)
        self.cursor = self.conn.cursor()
        self.open = True

        if create:
            self._create_table()
            if entities is not None:
                self.insert_new_formulas(entities)


    def _create_table(self):
        query = f'''
            CREATE TABLE IF NOT EXISTS {FORMULAS_TABLE} (
                {ENTITY} TEXT PRIMARY KEY NOT NULL,
                {FORMULA} CHAR(100)
            );
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def close(self):
        """
        Αν η σύνδεση με την βάση είναι ενεργή, την κλείνει και διαγράφει
        το λεξικό των εγγραφών που είναι φορτωμένα στη μνήμη.
        """
        if self.open:
            self.open = False
            self.cursor.close()
            self.conn.close()
            FormulasDB.in_memory_formulas = {}


    def _insert_formula(self, drug, save_new):
        """
        Ανακτά από το pubchem api και επιστρέφει το μοριακό τύπο της ουσία drug_name.
        :param drug_name: Το όνομα μιας ουσίας
        :param save_new:  True αν το νέο ζεύγος (ουσίας, τύπος) πρέπει να αποθηκευτεί
                          στη βάση.
        :return: Συμβολοσειρά με τον τύπο, αλλίως αν δεν βρέθηκε None
        """
        formula = self.req_molecular_formula(drug)
        if save_new:
            query = f'''
                INSERT INTO {FORMULAS_TABLE} ({ENTITY}, {FORMULA}) 
                VALUES (\'{drug}\', \'{formula}\');
            '''
            try:
                self.cursor.execute(query)
            except sqlite3.OperationalError:
                pass
        return formula


    def insert_new_formulas(self, entities):
        """
        Ανακτά μέσω του pubchem api τους μοριακούς τύπους των entities
        και τους αποθηκεύει στη βάση (δεν ελέγχει πρώτα αν υπάρχουν).
        Επίσης στο τέλος κλείνει τη σύνδεση με τη βάση.
        :param entities: Μοναδικές οντότητες
        """
        for drug in tqdm(entities):
            f = self._insert_formula(drug, save_new=True)
        self.conn.commit()
        self.close()


    def get_formula(self, drug, save_new=True):
        """
        :return: Τον μοριακό τύπο της ουσίας με όνομα w, αν υπάρχει.
                 Αλλιώς, αν δεν μπορεί να τον βρει επιστρέφει None.
        """
        """
        1. Αν έχει αποθηκευμένο τον ΜΤ στο λεξικό (η ουσία απαντήθηκε ξανά)
           τον επιστρέφει
        2. Αλλιώς, τον ανακτά μέσω του pubchem api
        3. Τον αποθηκεύει στο λεξικό για μελλοντική προσπέλαση
        4. Τον επιστρέφει  
        """
        formula = FormulasDB.in_memory_formulas.get(drug)

        if formula is None:
            try:
                query = f'SELECT {FORMULA} FROM {FORMULAS_TABLE} WHERE {ENTITY} = \'{drug}\''
                formula = self.cursor.execute(query).fetchone()[0]
            except TypeError:
                formula = self._insert_formula(drug, save_new)
                self.conn.commit()

            FormulasDB.in_memory_formulas[drug] = formula

        return formula


    def req_molecular_formula(self, drug_name):
        """
        Ανακτά από το pubchem api και επιστρέφει το μοριακό τύπο της ουσία drug_name.
        :param drug_name: Το όνομα μιας ουσίας
        :return: Συμβολοσειρά με τον τύπο, αλλίως αν δεν βρέθηκε ή αν υπήρξε κάποιο
                 σφάλμα επιστρέφει None
        """
        query = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + drug_name + \
                '/property/MolecularFormula/JSON'
        try:
            response = requests.get(query)
        except Exception:
            return None
        if response.status_code < 400:
            return response.json()['PropertyTable']['Properties'][0]['MolecularFormula']

        return None


    def different_formulas(self, w1, w2):
        """
        Ελέγχει αν δύο ουσίες έχουν διαφορετικό μοριακό τύπο => άρα είναι
        διακριτές.
        1. Ανακτά τους μοριακούς τύπους των δύο ουσιών
        2. Ελέγχει αν βρέθηκαν
        3. Αν βρέθηκαν και είναι διαφορετικοί μεταξύ τους,
           τότε οι ουσίες θεωρούνται διαφορετικές και επιστρέφει True
           (οι οντότητες είναι διακριτές)
        4. Αλλιώς αν δε βρέθηκε κάποιος μοριακός τύπος ή οι τύποι ήταν ίδιοι,
           δεν μπορεί να βγάλει συμπέρασμα αν οι οντότητες είναι διακριτές (False)
        """
        w1_formula = self.get_formula(w1)  # 1
        w2_formula = self.get_formula(w2)

        w1_found = w1_formula is not None  # 2
        w2_found = w2_formula is not None

        if w1_found and w2_found and w1_formula != w2_formula:  # 3
            return True
        return False  # 4



