import re

from other.CONSTANTS import *
from other.file_paths import processed_data_path
from source.database_pcg.formulas_db import FormulasDB
from source.database_pcg.synonyms_db import *
from source.sameAs.similarity_metrics import calc_similarity as sim, FZ, RATIO, COSINE, TOKEN_SORT_R


class SameAs:

    # Βάση μοριακών τύπων
    formulas = None

    synonyms_found = None

    def __init__(self, fz_type=RATIO, fz_cutoff=85, fz_thrs=95, sec_cr=None, sec_thrs=None, reset_synonyms=True):
        """
        :param fz_type: Ο τύπος fuzzy ομοιότητας που επιθυμείται
        :param fz_cutoff: Το κάτω όριο ομοιότητας που θα πρέπει να έχει ένα ζεύγος για να το εξετάσει
                          ως πιθανό ζεύγος sameAs
        :param fz_thrs:   Το κάτω όριο fuzzy ομοιότητας για να θεωρήσει ότι ένα ζεύγος είναι πραγματικά
                          sameAs
        :param sec_cr:    Ο τύπος του δευτερεύοντος κριτηρίου που θα εξεταστούν όλα τα ζεύγη με fuzzy
                          ομοιότητα μεταξύ fz_cutoff και fz_thrs για το αν είναι sameAs.
                          Σκοπεύει να εντοπίσει ζευγάρια που δεν πέτυχε το πρώτο κριτήριο.
        :param sec_thrs:  Το κάτω όριο για το δεύτερο κριτήριο, ώστε να θεωρηθεί ένα ζεύγος sameAs

        """
        if reset_synonyms or SameAs.synonyms_found is None:
            SameAs.synonyms_found = set()

        SameAs.formulas = FormulasDB()
        self.fz_type = fz_type
        self.fz_cutoff = fz_cutoff
        self.fz_thrs = fz_thrs
        self.sec_cr = sec_cr
        self.sec_thrs = sec_thrs
        self.reset_synonyms = reset_synonyms



    def join(self, ddi, other, rename=True):
        """
        Χρησιμοποιείται για τον συνδυασμό των αποτελεσμάτων πολλαπλών ρυθμίσεων αναζήτησης
        για sameAs ζεύγη. Θα βγάλει ζεύγη για το τρέχον και το other αντικείμενο και θα τα
        συνδικάσει απορρίπτοντας τα διπλότυπα. (x sameAs y) == (y sameAs x)
        :param ddi: Μια συλλογή οντοτήτων
        :param other:  Ένα άλλο αντικείμενο της κλάσης sameAs, που θα βρει ζεύγη sameAs με
                       βάση διαφορετικά κριτήρια (παραμέτρους) από το τρέχον αντικείμενο
        :param rename: True ώστε να επιστρέψει DF με πεδία με ονομασίες
                       <RDF_S=w1, RDF_P=SAME_AS, RDF_O=w2>
                       Αλλιώς <Ε1=w1, RDF_P=SAME_AS, Ε2=w2>
        :return: DF με τριπλέτες με predicate το SAME_AS. Τα ζεύγη των ονομάτων της συλλογής
                 που αναφέρονται στην ίδια οντότητα. Το σύνολο προκύπτει από την ένωση των
                 αποτελεσμάτων που δίνουν τα δύο αντικείμενα SameAs.
        """
        print('Running 1st...')
        df = self.get_sameAs_triples(ddi, rename=rename)
        print('Running 2nd...')
        df2 = other.get_sameAs_triples(ddi, rename=rename)

        # Έχει γίνει μετονομασία των πεδίων ή όχι
        columns = [RDF_S, RDF_O] if rename else [E1, E2]

        df = df.append(df2, ignore_index=True)
        df = df.loc[
            pd.DataFrame(np.sort(df[columns], 1), index=df.index)
                .drop_duplicates(keep='first').index
        ]
        df.reset_index(inplace=True, drop=True)
        SameAs.formulas.close()
        return df



    def get_sameAs_triples(self, ddi, rename=True):
        """
        Χρησιμοποιείται για την παραγωγή των ζευγών sameAs σύμφωνα με τα κριτήρια
        αναζήτησης που τέθηκαν στο τρέχον αντικείμενο
        :param ddi: Μια συλλογή οντοτήτων
        :param rename: True ώστε να επιστρέψει DF με πεδία με ονομασίες
                       <RDF_S=w1, RDF_P=SAME_AS, RDF_O=w2>
                       Αλλιώς <Ε1=w1, RDF_P=SAME_AS, Ε2=w2>
        :return: DF με τριπλέτες με predicate το SAME_AS. Τα ζεύγη των ονομάτων της συλλογής
                 που αναφέρονται στην ίδια οντότητα.
        """
        df = self._find_sameAs(ddi)
        df = df[df['y_pred'] == 1].reset_index(drop=True)
        if rename:
            df.rename(columns={E1: RDF_S, E2: RDF_O}, inplace=True)
            df[RDF_P] = SAME_AS
            return df[RDF_TRIPLE]
        return df


    def _find_sameAs(self, ddi):
        """
        Εφαρμόζει τα κριτήρια που τέθηκαν για να εξετάσει τα υποψήφια ζεύγη και να εντοπίσει
        για ποια από αυτά ισχύει η ιδιότητα sameAs.
        :param ddi: Μια συλλογή οντοτήτων
        :return: DF <E1=w1, E2=w2, y_pred= 0 ή 1>. Περιέχει όλα τα ζεύγη που εξέτασε και
                 έχουν ομοιότητα fz μεγαλύτερη ή ίση του κατωφλιού fz_cutoff. Ανάλογα με τα
                 αποτελέσματα των κριτηρίων που τεθήκαν, μπορεί w1 sameAs w2 (1) ή όχι (0).
        """
        """
        0. Αν είναι η πρώτη φορά που τρέχει την διαδικασία για το συγκεκριμένο σύνολο:
           Συγκεντρώνει τα ζεύγη των ουσιών που είναι συνώνυμα σύμφωνα με την βάση συνωνύμων
        1. Υποψήφια ζεύγη για έλεγχο: Ζεύγη με fuzzy ομοιότητα τύπου fz_type >= του
           fz_cutoff
        2. Για κάθε τέτοιο ζεύγος w1, w2 :
           3. Αν επιπλέον αυτή η ομοιότητα >= fz_thrs (fz_cutoff < fz_thrs <= ομοιότητα fz)
              => w1 sameAs w2 (1)
           4. Αλλιώς αν fz_cutoff <= ομοιότητα fz < fz_thrs, έλεγξε το 2ο κριτήριο sec_cr:
              5. Αν ομοιότητα sec_cr >= thrs_sec => w1 sameAs w2 (1)
                 αλλιώς w1 not sameAs w2 (0)
        
           6. Αν προέκυψε από τα παραπάνω ότι w1 sameAs w2 (1), πρέπει να ελεγχεί η ορθότητα
              αυτής της υπόθεσης :
              7. Αν w1, w2 έχουν διαφορετικούς μοριακούς τύπους αναιρείται
                 => w1 not sameAs w2 (0)
        8. Συγχωνεύει τα ζεύγη συνωνύμων με αυτά που προέκυψαν από τα κριτήρια ομοιότητας
        9. Επιστρέφει όλα τα ζεύγη που εξέτασε, μαζί με την κατηγοριοποίηση που τους έδωσε
        """
        unique_entities = ddi[E1].append(ddi[E2]).unique()
        if self.reset_synonyms :  # 0
            db = Synonyms()
            SameAs.synonyms_found = SameAs.synonyms_found\
                .union(db.find_collection_synonyms(set(unique_entities)))

        df = self.find_similar_pairs(unique_entities.tolist())  # 1

        for i in df.index:  # 2
            w1 = df[E1][i]
            w2 = df[E2][i]

            if df['sim'][i] + e >= self.fz_thrs:  # 3
                df.at[i, 'y_pred'] = 1

            # 2ο κριτήριο ομοιότητας
            elif self.sec_cr is not None:  # 4
                df.at[i, 'y_pred'] = 1 if self._second_cr(w1, w2) else 0  # 5

            # 3ο κριτήριο : μοριακός τύπος
            if df['y_pred'][i] == 1:  # 6
                if self.different_formulas(w1, w2):  # 7
                    df.at[i, 'y_pred'] = 0

        if self.reset_synonyms:  # 8
            df_from_synonyms = pd.DataFrame(SameAs.synonyms_found, columns=[E1, E2])
            df_from_synonyms['sim'] = 1
            df_from_synonyms['y_pred'] = 1
            df = pd.concat([df, df_from_synonyms]).reset_index(drop=True)

        return df  # 9


    def _second_cr(self, w1, w2):
        """
        Υπολογίζει την ομοιότητα/απόσταση συμφωνά με το όρισμα για το 2ο κριτήριο
        και επιστρέφει True αν είνα μεγαλύτερη από το κάτω όριο thrs_sec που τέθηκε.
        Τότε το ζεύγος είναι υποψήφιο για sameAs.
        """
        return sim(w1, w2, sim_metric=self.sec_cr) + e > self.sec_thrs



    def find_similar_pairs(self, entities):
        """
        Εντοπίζει τα ζεύγη των ονομάτων των οντοτήτων της συλλογής που αποτελούν
        πιθανά sameAs και πρέπει να εξεταστούν, έχουν δηλ fuzzy ομοιότητα μεγαλύτερη
        του κατωτέρου κατωφλιού fz_cutoff
        :param entities: Λίστα με τα μοναδικά ονόματα των οντοτήτων μιας συλλογής,
                         όπως του ddi.
        :return: DF με <E1, Ε2 : ονόματα οντοτήτων, sim : fuzzy ομοιότητα τους>
                 Περιλαμβάνει τα ζεύγη που έχουν sim >= fz_cutoff και δεν καταπατούν
                 τον κανόνα των prefixes
        """
        """
        1. Για κάθε ζεύγος ονομάτων w1, w2 (κάτω τριγωνικός πίνακας):
           2. Αν το ζεύγος υπάρχει ήδη από τα συνώνυμα, το προσπερνάει 
           3. Υπολόγιζει τη fuzzy ομοιότητα τους σύμφωνα με τον τύπο που έχει οριστεί
              και την κράτα σε 2d dictionary {w1 : {... w2 : ομοιότητα(w1,w2)...} }
           5. Αν η ομοιότητα είναι κάτω από το κατώτατο κατώφλι ή το ζεύγος δεν ικανοποιεί
              τον κανόνα των prefixes :
              5. τότε το ζεύγος δεν είναι sameAs και απορρίπτεται
        6. Δημιουργεί DF <E1=w1, E2=w2, sim=ομοιότητα(w1,w2)>
        7. Απορρίπτει τις εγγραφές όπου w1==w2, ζεύγη του πάνω τμήματος του πίνακα που δεν
           υπολογίστηκε ομοιότητα, καθώς και ζεύγη που απορρίφθηκαν στο 3
        8. Ταξινομεί τα ζεύγη συμφωνά με την ομοιότητα τους       
        """
        df = {}

        for i in range(0, len(entities)):  # 1
            w1 = entities[i]
            for j in range(i, len(entities)):
                w2 = entities[j]

                if w1 not in df:
                    df[w1] = {}

                if (w1, w2) in SameAs.synonyms_found or (w2, w1) in SameAs.synonyms_found:  # 2
                    continue

                df[w1][w2] = sim(w1, w2, sim_metric=FZ, fz_type=self.fz_type)  # 3

                if df[w1][w2] < self.fz_cutoff or self.different_prefix(w1, w2):  # 4
                    df[w1][w2] = 0  # 5

        df = pd.DataFrame(df)  # 6
        df[E1] = entities
        df = df.melt(id_vars=E1, var_name=E2, value_name='sim')
        df = df[(df[E1] != df[E2]) & (df['sim'].notna()) & (df['sim'] != 0)]  # 7

        df.sort_values('sim', inplace=True, ignore_index=True, ascending=[False])  # 8
        return df


# ----------------------------------------------------------------------------------------------


    def different_formulas(self, w1, w2):
        """
        Ελέγχει αν δύο ουσίες έχουν διαφορετικό μοριακό τύπο => άρα είναι διακριτές.
         - Αν βρέθηκαν και είναι διαφορετικοί μεταξύ τους, τότε οι ουσίες θεωρούνται διαφορετικές
           και επιστρέφει True (οι οντότητες είναι διακριτές)
         - Αλλιώς αν δε βρέθηκε κάποιος μοριακός τύπος ή οι τύποι ήταν ίδιοι,
           δεν μπορεί να βγάλει συμπέρασμα αν οι οντότητες είναι διακριτές (False)
        """
        return SameAs.formulas.different_formulas(w1, w2)


    def formula(self, w):
        """
        :return: Τον μοριακό τύπο της ουσίας με όνομα w, αν υπάρχει.
                 Αλλιώς, αν δεν μπορεί να τον βρει επιστρέφει None.
        """
        return SameAs.formulas.get_formula(w)


# ---------------------------------------------------------------------------------------

    def different_prefix(self, w1, w2):
        """
        Ορίζει κανόνα για τον έλεγχο των προθεμάτων δύο ονομάτων.
        :return: False αν το ζεύγος είναι υποψήφιο sameAs
                 'non-steroid ...' με 'nonsteroid ...'
                 'aspirin' με  'non-steroid ...'
                 ή οποιοδήποτε άλλο απλό ζεύγος

                 True αν πρόκειται για λέξεις τουλ μία έχει πρόθεμα
                 ενδιαφέροντος, η άλλη έχει διαφορετικό πρόθεμα και
                 έχουν ίδιο tail => Διακριτές οντότητες, δεν πρέπει
                 να ελεγχθούν.
                 'tricyclic ...' με 'tetracyclic ...'
        """
        prefixes = re.compile(r'(non|tri|tetra)')
        p = re.compile(r'^(non|tri|tetra)'
                       r'[\s-]*')

        def parse_word(w):
            """
            Βρίσκει την πρώτη λέξη μέσα στο όνομα μαζί με το πρόθεμα.
            Αν η λέξη ξεκινά από κάποιο από τα προθέματα ενδιαφέροντος,
            την σπάει σε prefix + tail (υπόλοιποι χαρακτήρες της 1ης λέξης)
            'non-steroid' -> non + steroid
            'tricyclic' -> tri + cyclic
            'aspirin' -> None + aspirin
            """
            prefix_match_w = [m for m in p.finditer(w)]

            if len(prefix_match_w) != 0:
                prefix_match_w = prefix_match_w[0]
                prefix = prefix_match_w.group(0)
                prefix = prefixes.findall(prefix)[0]
                prefix_end = prefix_match_w.end()
            else:
                prefix = None
                prefix_end = 0

            # w[prefix_end:] -> Το υπόλοιπο μετά το prefix
            # Διαχωριστικά μεταξύ των λέξεων ' ' ή '-'
            tail = re.split(' |-', w[prefix_end:])[0]

            return prefix, tail
        # ---------------------------------
        # Σπάει τις λέξεις σε prefix + tail
        w1_prefix, w1_tail = parse_word(w1)
        w2_prefix, w2_tail = parse_word(w2)

        """
        Αν έχουν ίδιο tail (steroid) πρέπει να έχουν και ίδιο prefix (non ή None),
        για να θεωρηθούν πιθανό sameAs pair. Αλλιώς το ζεύγος απορρίπτεται 
        (non + steroid με None + steroid)
        """
        if w1_tail == w2_tail:
            if w1_prefix == w2_prefix:
                return False
            else:
                return True
        return False


# -----------------------------------------------------------------------------------

def load_sameAs(ddi, train):
    file = processed_data_path(train) + 'sameAs.csv'
    try:
        sameAs = pd.read_csv(file)

    except FileNotFoundError:  # αν δεν έχει φτιαχτεί το αρχείο, το φτιάχνει από το ddi

        s1 = SameAs(fz_type=RATIO, fz_cutoff=85, fz_thrs=95, sec_cr=COSINE, sec_thrs=0.965, reset_synonyms=True)
        s2 = SameAs(fz_type=TOKEN_SORT_R, fz_cutoff=85, fz_thrs=94, sec_cr=COSINE, sec_thrs=0.965, reset_synonyms=False)
        sameAs = s1.join(ddi, s2)
        sameAs.to_csv(file, sep=',', encoding='utf-8', index=False)
    return sameAs










