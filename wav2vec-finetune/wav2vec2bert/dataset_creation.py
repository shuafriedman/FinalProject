import re
from config import *
from datasets import load_dataset, dataset_dict, concatenate_datasets

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'



def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def get_vocab(dataset):
        vocab = dataset.map(extract_all_chars, batched=True, batch_size=-1, \
                        keep_in_memory=True, remove_columns=dataset.column_names)
        vocab_list = list(set(vocab["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        return vocab_dict
    
def get_train_test_vocab(dataset):
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

def standardize_dataset(dataset):
    dataset= dataset.remove_columns([col for col in dataset.features if col \
                                not in ["sentence", "transcription", "audio"]])
    try:
        dataset= dataset.rename_column("sentence", "transcription")
    except:
        pass
    dataset= dataset.map(lambda x: {'dataset': dataset})
    dataset= dataset.map(remove_special_characters)
    return dataset
def main():
    datasets = []
    for dataset_config in DATASETS:
        print("Loading dataset: ", dataset_config['name'])
        path = dataset_config['name'] if FROM_HUB else dataset_config['local_path']
        split = None if TRAIN_AND_TEST else dataset_config['test_split'] #none grabs both train and test
        dataset = load_dataset(path=path,
                                name=dataset_config['language'],
                                split=split,              
        )
        print("Standardizing dataset: ", dataset_config['name'])
        if TRAIN_AND_TEST:
            print("Standardizing train and test splits")
            for name, data in dataset.items():
               dataset[name] = standardize_dataset(data)
        else:
            print("Standardizing test split")
            dataset = standardize_dataset(dataset)
            dataset['test'] = dataset[dataset_config['test_split']]
        datasets.append(dataset)
        
    if TRAIN_AND_TEST:
        dataset = {'train': [], 'test':[]}
        for data in datasets:
            dataset['train'].append(data['train'])
            dataset['test'].append(data['test'])
        print("Concatenating datasets")
        dataset['train']= concatenate_datasets(dataset['train'])
        dataset['test']= concatenate_datasets(dataset['test'])
        vocab = get_train_test_vocab(dataset)
    print("Getting Vocab")
    vocab = get_vocab(dataset)

    # dataset = load_dataset("google/fleurs", "he_il", split="test")
    # kan_fleurs= kan_fleurs.map(remove_special_characters)
    
if __name__=='__main__':
    main()