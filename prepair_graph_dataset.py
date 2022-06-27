from main import make_db, run_pipeline
from other.utils import check_folders
from source.classification_task.dataset_preparation.graph_dataset import GraphDataset

"""
Τρέχει όλο το pipeline μέχρι την δημιουργία των datasets για την εκπαίδευση του GNN
(μέχρι και την εφαρμογή του BERT δηλαδή).
Αν επιθυμείται μόνο η εκτέλεση τους παραγωγής των datasets και όχι όλου του pipeline
να οριστεί NEO4J_RESET = False στο config.py
"""

def main():
    run_make_db = check_folders()
    if run_make_db :
        make_db()
    for train in [False, True]:
        run_pipeline(train=train)
        print(GraphDataset(train=train, add_inverse=False).graph)



if __name__ == "__main__":
    main()
