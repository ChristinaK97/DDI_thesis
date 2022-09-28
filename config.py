# ==================================================================================
#                   Εδώ ορίζονται τιμές μεταβλητών από το χρήστη
# ==================================================================================

# Το path όπου έχει τοποθετηθεί ο φάκελος του project. Ο καθορισμός του απαιτείται.
# Προσοχή: Να καταλήγει σε /
PROJECT_PATH = 'C:/Users/xristina/Desktop/progr/Python/DDI_thesis/'

# Αν θα ξαναφτιάξει τα απαραίτητα αρχεία για τη διαδικασία εύρεσης συνωνύμων -> True
# (σημειώνεται ότι η διαδικασία είναι χρονοβόρα λόγω κλήσεων σε API)
# ή αν θα χρησιμοποιήσει τα υπάρχοντα ή θα προσπαθήσει να τα κατεβάσει -> False
RESET_SYNONYMS_DB = False

# ================================ NEO4j =============================================

# Το path όπου έχει τοποθετηθεί ο φάκελος του neo4j
# Προσοχή: Να καταλήγει σε /
NEO4J_PATH = 'C:/neo4j/'

# Όνομα χρήστη και password που δόθηκε στο neo4j
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "290197"

# Αν θα ξαναφορτώσει τη συλλογή στο neo4j (μπορεί γίνει επανεκτέλεση ακόμα και αν φτιαχτεί προηγουμένως) -> True
# ή αν έχει φτιαχτεί προηγουμένως και δεν επιθυμείται επανεκτέλεση -> False
NEO4J_RESET = True

# Ονόματα βάσεων για κάθε σετ (δεν απαιτείται τροποποίηση)
NEO4J_TRAIN_DB = 'ddidb.train'
NEO4J_TEST_DB = 'ddidb.test'

# ============================== Εκπαίδευση μοντέλου ===================================

# --------------------------------- Επιλογή BERT --------------------------------------

bert_base = 'bert-base-cased'
biobert_base = 'monologg/biobert_v1.1_pubmed'
biobert_large = 'dmis-lab/biobert-large-cased-v1.1'
scibert = 'allenai/scibert_scivocab_cased'

cache_path = 'C:/Users/xristina/.cache/huggingface/transformers/aligned/aligned-lm/'

biobert_lbl = 'biobert/lbl'
biobert_par = 'biobert/par'
biobert_lbl_monologg_10epochs = 'biobert/lbl-monologg-10epochs'

scibert_lbl_3ep = 'scibert/lbl/3epochs'
scibert_lbl_10ep = 'scibert/lbl/10epochs'
scibert_par = 'scibert/par'

BERT_MODEL_NAME = cache_path + scibert_par

# ----------------------------- Παράμετροι εκπαίδευσης ---------------------------------

# Ρυθμός μάθησης
LR = 0.00005

# Weight decay για το Adam optimizer
WEIGHT_DECAY = 5 * 1e-4

# Πλήθος εποχών εκπαίδευσης
EPOCHS = 171

# Μέγεθος συνόλου επικύρωσης. Πρέπει να είναι μεταξύ 0 και 1
# (πχ 0.3 -> 30% του αρχικού συνόλου εκπαίδευσης θα χρησιμοποιηθεί ως επικύρωσης).
# Αν 0 τότε ολόκληρο το σύνολο εκπαίδευσης θα χρησιμοποιηθεί για εκπαίδευση
VAL_PREC = 0

# Εκτύπωση στατιστικών λανθασμένης ταξινόμησης
# Επιπλέον, παραγωγή αρχείου PROJECT_PATH/data/models/misclassified.csv για εξέταση
PRINT_MISCLS = True

# ----------------------------- Παράμετροι μοντέλου ---------------------------------

# Για evaluation ενός αποθηκευμένου μοντέλου στο test set -> 'INFERENCE'
# (αν δε βρεθεί αποθηκευμένο μοντέλο η εκπαίδευση θα εκτελεστεί και θα αποθηκευτεί το μοντέλο)
# Ή για εκπαίδευση μοντέλου -> 'TRAINING'
MODE = 'TRAINING'
MODEL_FILE   = PROJECT_PATH + 'data/models/classification_model.bin'
MODEL_CONFIG_FILE = PROJECT_PATH + 'data/models/classification_model_config.json'

# Πλήθος νευρώνων σε κάθε επίπεδο προεπεξεργασίας
# πχ [256] ένα επίπεδο προεπεξεργασίας με 256 νευρώνες
#    None : κανένα επίπεδο προεπεξεργασίας
MLP_PREPROCESSING_DIM = [256]

# Όμοια για μετεπεξεργασίας
MLP_POSTPROCESSING_DIM = [128, 64]

# Τελεστής GNN που θα εφαρμοστεί πχ 'SAGEConv', 'GINConv', 'GATConv'
GNN_TYPE = 'GINConv'

# Διάσταση διανυσματικού χώρου του GNN
HIDDEN_CHANNELS = 256

# Συνάρτηση ενεργοποίησης πχ 'relu', 'leaky_relu', 'tanh'
ACTIVATION_FUNC = 'relu'
