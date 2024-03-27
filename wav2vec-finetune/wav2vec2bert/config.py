fleurs_config = {
    'name': "google/fleurs",
    'local_path': 'fleurs',
    'language': 'he_il',
    'test_split': 'test'
}
ivritai_config = {
    'name': "ivrit-ai/whisper-training",
    'local_path': 'ivritai',
    'language': None,
    'test_split': 'test'
}

BASE_MODEL_NAME= "facebook/w2v-bert-2.0"
TRAIN_AND_TEST = True
DATASETS = [ivritai_config]
KEEP_HEBREW_ONLY=True
FROM_HUB = False
FINETUNED_MODEL_PATH = f"models/{BASE_MODEL_NAME}-finetuned"
DATA_FOLDER_PATH = "datasets"
LOAD_DATASET_FROM_LOCAL=False
DOWNLOAD_MODEL_LOCALLY=True
LOCAL_MODEL_PATH=f"models/{BASE_MODEL_NAME}"
DRY_RUN=True