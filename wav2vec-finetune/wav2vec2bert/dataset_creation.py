import re
from config import *
from datasets import load_dataset, concatenate_datasets, DatasetDict
import json
from pathlib import Path
import string
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
# hebrew_letters = [
#     'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט',
#     'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ',
#     'ק', 'ר', 'ש', 'ת'
# ]
# Using enumerate to pair each letter with an index, starting with 1
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
                                not in ["sentence", "transcription", "audio"]])
    try:
        dataset= dataset.rename_column("sentence", "transcription")
    except:
        pass
    print("writing name of dataset to dataset column")
    dataset= dataset.map(lambda x: {'dataset': dataset_name})
    dataset= dataset.map(remove_special_characters)
    return dataset

def main():
    #check if MODEL_FOLDER_NAME exists, wher current file is being invoke
    if not Path(MODEL_FOLDER_PATH).exists():
        Path(MODEL_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    
    datasets = []
    for dataset_config in DATASETS:
        print("Loading dataset: ", dataset_config['name'])
        path = dataset_config['name']
        split = None if TRAIN_AND_TEST else dataset_config['test_split'] #none grabs both train and test
        dataset = load_dataset(path=path,
                                name=dataset_config['language'],
                                split=split,              
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
    if KEEP_HEBREW_ONLY:
        for name in dataset:
            dataset[name] = drop_english_samples(dataset[name])
    vocab = get_vocab(dataset)
    vocab["|"]= vocab[" "]
    del vocab[" "]
    vocab["[UNK]"]= len(vocab)
    vocab["[PAD]"]= len(vocab)
    with open(f"{MODEL_FOLDER_PATH}/vocab.json", "w") as f:
        json.dump(vocab, f)
    
    dataset.save_to_disk(f"{DATA_FOLDER_PATH}/fleurs")
    # dataset = load_dataset("google/fleurs", "he_il", split="test")
    # kan_fleurs= kan_fleurs.map(remove_special_characters)
    
if __name__=='__main__':
    main()