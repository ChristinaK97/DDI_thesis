from config import MODE
from source.classification_task.gnn_models.train_classification_model import TrainClassificationModel, INFERENCE, \
    TRAINING


if __name__ == "__main__":
    TrainClassificationModel(mode=INFERENCE if MODE == 'INFERENCE' else TRAINING)

