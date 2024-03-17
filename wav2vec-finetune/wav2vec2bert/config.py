fleurs_config = {
    'name': "google/fleurs",
    'local_path': 'fleurs',
    'language': 'he_il',
    'test_split': 'test'
}
TRAIN_AND_TEST = True
DATASETS = [fleurs_config]
FROM_HUB = True
MODEL_FOLDER_PATH = "models/wav2vec2BertLm"
DATA_FOLDER_PATH = "datasets"