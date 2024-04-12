import re
from config import *
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Audio
import json
from pathlib import Path
import string
from transformers import Wav2Vec2BertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertProcessor
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\]\[\{\}\־]'
# hebrew_letters = [
#     'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט',
#     'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ',
#     'ק', 'ר', 'ש', 'ת'
# ]
# Using enumerate to pair each letter with an index, starting with 1
def prepare_dataset(batch, processor, input_key="input_features"):
    audio = batch["audio"]
    if input_key=="input_features":
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

def filter_long_samples(dataset):
    def is_shorter_than_max_duration(example):
        duration_seconds = len(example['array']) / example['sampling_rate']
        return duration_seconds <= FILTER_THRESHOLD
    filtered_dataset=dataset.filter(lambda example: is_shorter_than_max_duration(example['audio']))
    return filtered_dataset

def subsample_dataset(dataset):
    assert 0 < SUBSAMPLE_RATIO <= 1, "Subsample ratio must be between 0 and 1."
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * SUBSAMPLE_RATIO)))

def drop_english_samples(dataset):
    def contains_english_or_digits(text):
        english_letters = set(string.ascii_lowercase)
        digits = set(string.digits)
        return any(char in english_letters or char in digits for char in text.lower())
    filtered_dataset = dataset.filter(lambda example: not contains_english_or_digits(example['transcription']))
    return filtered_dataset

def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}
    
def get_vocab(dataset):
    vocabs = {}
    for split, data in dataset.items():
        vocabs[split]=data.map(extract_all_chars, batched=True, batch_size=-1, \
                        keep_in_memory=True, remove_columns=data.column_names)
    vocab = list(set(vocabs['train']["vocab"][0]) | set(vocabs['test']["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    return vocab_dict
        
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

def standardize_dataset(dataset, dataset_name):
    print("Removing unecessary columns")
    dataset= dataset.remove_columns([col for col in dataset.features if col \
                                not in ["sentence", "transcription", "text", "audio"]])
    try:
        dataset= dataset.rename_column("sentence", "transcription")
    except:
        pass
    try:
        dataset = dataset.rename_column("text", "transcription")
    except:
        pass
    print("writing name of dataset to dataset column")
    dataset= dataset.map(lambda x: {'dataset': dataset_name})
    dataset= dataset.map(remove_special_characters)
    return dataset

def main():
    # check if MODEL_FOLDER_NAME exists, wher current file is being invoke
    if not Path(LOCAL_MODEL_PATH).exists():
        Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
    if not Path(FINETUNED_MODEL_PATH).exists():
        Path(FINETUNED_MODEL_PATH).mkdir(parents=True, exist_ok=True)

    datasets = []
    for dataset_config in DATASETS:
        print("Loading dataset: ", dataset_config['name'])
        path = dataset_config['name']
        split = None if TRAIN_AND_TEST else dataset_config['test_split'] #none grabs both train and test
        if LOAD_DATASET_FROM_LOCAL:
            dataset = load_from_disk(
                dataset_config['local_path']
                )
        else:
            dataset = load_dataset(
                path=path,
                name=dataset_config['language'],
                split=split             
            )
        print("Standardizing dataset: ", dataset_config['name'])
        if TRAIN_AND_TEST:
            print("Standardizing train and test splits")
            for name, data in dataset.items():
               dataset[name] = standardize_dataset(data, dataset_config["name"])
        else:
            print("Standardizing test split")
            dataset = standardize_dataset(dataset, dataset_config["name"])
            dataset = dataset.train_test_split(test_size=0.2)
        
        datasets.append(dataset)

    if len(datasets) > 1:
        print("Concatenating datasets")    
        dataset = {'train': [], 'test':[]}
        for data in datasets:
            dataset['train'].append(data['train'])
            dataset['test'].append(data['test'])
        print("Concatenating datasets")
        dataset['train']= concatenate_datasets(dataset['train'])
        dataset['test']= concatenate_datasets(dataset['test'])
        dataset = DatasetDict({
            'train': dataset['train'],
            'test': dataset['test']
        })
    if SUBSAMPLE_RATIO < 1.0:
        for name in dataset.keys():
            dataset[name] = subsample_dataset(dataset[name])
    if FILTER_LONG_SAMPLES:
        for name in dataset:
            dataset[name] = filter_long_samples(dataset[name])
    if KEEP_HEBREW_ONLY:
        for name in dataset:
            dataset[name] = drop_english_samples(dataset[name])
            
    print("Casting Audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("Getting Vocab")
    vocab = get_vocab(dataset)
    vocab["|"]= vocab[" "]
    del vocab[" "]
    vocab["[UNK]"]= len(vocab)
    vocab["[PAD]"]= len(vocab)
    vocab["<s>"] = len(vocab)
    vocab["</s>"]=len(vocab)
    print("printing vocab with length: ", len(vocab))
    with open(f"{FINETUNED_MODEL_PATH}/vocab.json", "w") as f:
        json.dump(vocab, f)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(FINETUNED_MODEL_PATH,
                unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    if BASE_MODEL_NAME=="facebook/w2v-bert-2.0":
        input_key="input_features"
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
        processor=Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    else:
        input_key="input_features"
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, 
                padding_value=0.0, do_normalize=True, return_attention_mask=True)
        processor=Wav2Vec2Proessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if PERFORM_PREPROCESSING_ON_DATASET_CREATION:
        dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].features.keys(), \
                            fn_kwargs={"processor": processor, "input_key": input_key})
                        
    print("Saving dataset to disk")
    filtered_suffix = "-filtered" if FILTER_LONG_SAMPLES else None
    preprocessing_suffix="-proc" if PERFORM_PREPROCESSING_ON_DATASET_CREATION else None
    dataset.save_to_disk(f"{DATA_FOLDER_PATH}/{dataset_config['local_path']}{filtered_suffix}{preprocessing_suffix}")
    print("Downloading Base Model locally")
    if DOWNLOAD_MODEL_LOCALLY:
        print("downloading model")
        print("download feature extractor")

        model = Wav2Vec2BertForCTC.from_pretrained(BASE_MODEL_NAME, vocab_size=len(processor.tokenizer))
        print("download tokenizer")
        print("saving locally")
        model.save_pretrained(LOCAL_MODEL_PATH)
        processor.save_pretrained(LOCAL_MODEL_PATH)
        print("Finished saving")
    # dataset = load_dataset("google/fleurs", "he_il", split="test")
    # kan_fleurs= kan_fleurs.map(remove_special_characters)
    
if __name__=='__main__':
    main()