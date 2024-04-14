from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from transformers import SeamlessM4TFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor, AutoFeatureExtractor, AutoTokenizer, AutoModel

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
wav2vec2_config={
    "feature_extractor": AutoFeatureExtractor,
    "tokenizer": AutoTokenizer,
    "processor": Wav2Vec2Processor,
    "model": AutoModelForCTC,
    "model_name": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "input_key": "input_values"
}
DATA_FOLDER_PATH = "datasets"
LOAD_DATASET_FROM_LOCAL=False
DOWNLOAD_MODEL_LOCALLY=False
FILTER_LONG_SAMPLES = True
PERFORM_PREPROCESSING_ON_DATASET_CREATION = True
FILTER_THRESHOLD = 30
SUBSAMPLE_RATIO=1.0
USE_TRAINER=True
MODEL_CONFIG=wav2vec2_config
TRAIN_AND_TEST = True
DATASETS = [fleurs_config]
KEEP_HEBREW_ONLY=True
FROM_HUB = False
LOCAL_MODEL_PATH=f"models"
FINETUNED_MODEL_PATH = f"models"
DRY_RUN=True
MANUALLY_SET_MODEL_CONFIG=True