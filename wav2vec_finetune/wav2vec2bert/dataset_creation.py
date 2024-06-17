import re
from config import *
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Audio
import json
from pathlib import Path
import string
from transformers import Wav2Vec2BertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModelForCTC
from transformers import Wav2Vec2CTCTokenizer
from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertProcessor
import torch
import numpy as np
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\]\[\{\}\־]'
# hebrew_letters = [
#     'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט',
#     'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ',
#     'ק', 'ר', 'ש', 'ת'
# ]
# Using enumerate to pair each letter with an index, starting with 1
def prepare_dataset(batch, processor, input_key):
    audio = batch["audio"]
    if input_key=="input_features":
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    elif input_key=="input_values":
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

def remove_nan_batches(batch, input_key):
    if np.isnan(batch[input_key]).any():
        return False
    if np.isnan(batch['labels']).any():
        return False
    return True

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
    local_model_path =f"{LOCAL_MODEL_PATH}/{MODEL_CONFIG['model_name']}"
    finetuned_model_path= f"{FINETUNED_MODEL_PATH}/{MODEL_CONFIG['model_name']}-finetuned"
    print("Model Path")
    print("Finetuned Path")
    print(finetuned_model_path)
    # check if MODEL_FOLDER_NAME exists, wher current file is being invoke
    # if not Path(model_path).exists():
    #     Path(model_path).mkdir(parents=True, exist_ok=True)
    if not Path(finetuned_model_path).exists():
        Path(finetuned_model_path).mkdir(parents=True, exist_ok=True)

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
    with open(f"{finetuned_model_path}/vocab.json", "w") as f:
        json.dump(vocab, f)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(finetuned_model_path,
                unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor= MODEL_CONFIG['feature_extractor'].from_pretrained(MODEL_CONFIG['model_name'])
    processor=MODEL_CONFIG['processor'](feature_extractor=feature_extractor, tokenizer=tokenizer)
    print("saving processor")
    processor.save_pretrained(local_model_path + '-finetuned')
    if PERFORM_PREPROCESSING_ON_DATASET_CREATION:
        dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].features.keys(), \
                            fn_kwargs={"processor": processor, "input_key": MODEL_CONFIG['input_key']})
    for name in dataset:
        dataset[name] = dataset[name].filter(remove_nan_batches, fn_kwargs={"input_key": MODEL_CONFIG['input_key']})

    print("Saving dataset to disk")
    filtered_suffix = "-filtered" if FILTER_LONG_SAMPLES else None
    preprocessing_suffix="-proc" if PERFORM_PREPROCESSING_ON_DATASET_CREATION else None
    dataset.save_to_disk(f"{DATA_FOLDER_PATH}/{dataset_config['local_path']}{filtered_suffix}{preprocessing_suffix}")
    print("Downloading Base Model locally")
    if DOWNLOAD_MODEL_LOCALLY:
        print("downloading model")
        model = MODEL_CONFIG['model'].from_pretrained(MODEL_CONFIG['model_name'], vocab_size=len(processor.tokenizer))
        print("saving locally")
        model.save_pretrained(local_model_path)
        print("Finished saving")
    # dataset = load_dataset("google/fleurs", "he_il", split="test")
    # kan_fleurs= kan_fleurs.map(remove_special_characters)
    
if __name__=='__main__':
    main()