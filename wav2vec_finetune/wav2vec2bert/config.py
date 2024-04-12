fleurs_config = {
    'name': "google/fleurs",
    'local_path': 'fleurs-imvladikon', #todo remove imvladikon
    'language': 'he_il',
    'test_split': 'test'
}
ivritai_config = {
    'name': "ivrit-ai/whisper-training",
    'local_path': 'ivritai',
    'language': None,
    'test_split': 'test'
}
LOAD_DATASET_FROM_LOCAL=False
DOWNLOAD_MODEL_LOCALLY=False
FILTER_LONG_SAMPLES = True
PERFORM_PREPROCESSING_ON_DATASET_CREATION = True
FILTER_THRESHOLD = 30
SUBSAMPLE_RATIO=1.0
BASE_MODEL_NAME= "facebook/w2v-bert-2.0"
TRAIN_AND_TEST = True
DATASETS = [fleurs_config]
KEEP_HEBREW_ONLY=True
FROM_HUB = False
DATA_FOLDER_PATH = "datasets"
LOCAL_MODEL_PATH=f"models/{BASE_MODEL_NAME}"
FINETUNED_MODEL_PATH = f"models/{BASE_MODEL_NAME}-finetuned"
DRY_RUN=False
MANUALLY_SET_MODEL_CONFIG=True