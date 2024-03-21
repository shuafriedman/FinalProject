fleurs_config = {
    'name': "google/fleurs",
    'local_path': 'fleurs',
    'language': 'he_il',
    'test_split': 'test'
}
BASE_MODEL_NAME= "facebook/w2v-bert-2.0"
TRAIN_AND_TEST = False
DATASETS = [fleurs_config]
KEEP_HEBREW_ONLY=True
FROM_HUB = False
FINETUNED_MODEL_PATH = f"models/{BASE_MODEL_NAME}-finetuned"
DATA_FOLDER_PATH = "datasets"
DOWNLOAD_MODEL_LOCALLY=True
LOCAL_MODEL_PATH=f"models/{BASE_MODEL_NAME}"
DRY_RUN=True