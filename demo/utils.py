import re
import os
import json
from pathlib import Path
import string
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
SUBSAMPLE_RATIO = 1.0
FILTER_THRESHOLD = 15
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\]\[\{\}\־]'
# hebrew_letters = [
#     'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט',
#     'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ',
#     'ק', 'ר', 'ש', 'ת'
# ]
# Using enumerate to pair each letter with an index, starting with 1
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

def add_special_characters(vocab):
    vocab["|"]= vocab[" "]
    del vocab[" "]
    vocab["[UNK]"]= len(vocab)
    vocab["[PAD]"]= len(vocab)
    vocab["<s>"] = len(vocab)
    vocab["</s>"]=len(vocab)
    return vocab  
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

def standardize_dataset(dataset):
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
    print("Removing Special Characters")
    return dataset

def prepare_dataset(batch, processor, input_key):
    audio = batch["audio"]
    if input_key=="input_features":
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    elif input_key=="input_values":
        batch["input_values"] = processor(audio["array"].tolist(), sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))
