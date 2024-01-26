
# %pip install transformers datasets accelerate evaluate jiwer librosa soundfile
# %pip install accelerate>=0.25.0
import subprocess
import sys
# subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers', 'datasets', 'accelerate', 'evaluate', 'jiwer', 'librosa', 'soundfile'])
# subprocess.run([sys.executable, '-m', 'pip', 'install', 'accelerate>=0.25.0'])
import os
from datasets import load_dataset, IterableDatasetDict, load_from_disk
import os
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import Audio
import torch
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import numpy as np
from IPython.display import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def get_dataset(dataset_name, subset: float=1.0, 
                path: str=os.getcwd(),
                split_names: list= ['train', 'validation'],
                save_dataset: bool = True):
    
    dataset_path = path + 'datasets/' + dataset_name
    raw_datasets = IterableDatasetDict()
    print("dataset path: ", dataset_path)
    if os.path.exists(dataset_path)==False:
        print('getting dataset from huggingface')
        raw_datasets["train"] = load_dataset(dataset_name, split=split_names[0])  # set split="train+validation" for low-resource
        raw_datasets["test"] = load_dataset(dataset_name, split=split_names[1]) #no test for imvladikon hebrew
        if save_dataset:
            raw_datasets["train"].save_to_disk(dataset_path + "/train")
            raw_datasets["test"].save_to_disk(dataset_path + "/test")
    else:
        print("loading local data")
        raw_datasets["train"] = load_from_disk(dataset_path + "/train")
        raw_datasets["test"] = load_from_disk(dataset_path + "/test")
    if subset < 1.0:
        train_rows = raw_datasets["train"].num_rows
        test_rows = raw_datasets["test"].num_rows
        
        raw_datasets["train"] = raw_datasets["train"].shuffle().select(
            range(int(train_rows * subset))
        )
        raw_datasets["test"] = raw_datasets["test"].shuffle().select(
            range(int(test_rows * subset))
        )
    print(raw_datasets)
    return raw_datasets

def get_model_and_configs(name, path: str=None):
    model_path = path + 'models/' + name
    if path and os.path.isdir(model_path):
        try:
            model = WhisperForConditionalGeneration.from_pretrained(name)
            tokenizer = WhisperTokenizer.from_pretrained(name, language="Hebrew", task="transcribe")
            feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
            processor = WhisperProcessor.from_pretrained(name, language="Hebrew", task="transcribe")
        #except if path is not a directory
        except OSError:
            print("error getting model")
    else:
        print("getting model from hub")
        model = WhisperForConditionalGeneration.from_pretrained(name)
        tokenizer = WhisperTokenizer.from_pretrained(name, language="Hebrew", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
        processor = WhisperProcessor.from_pretrained(name, language="Hebrew", task="transcribe")
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        feature_extractor.save_pretrained(model_path)
        processor.save_pretrained(model_path)
    return model, tokenizer, feature_extractor, processor

def prepare_dataset(raw_datasets, processor):
    processor=processor
    def prep_batch(batch):
        raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
        do_lower_case = False
        do_remove_punctuation = False
        normalizer = BasicTextNormalizer()

        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]
        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        # optional pre-processing steps
        transcription = batch["sentence"]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = normalizer(transcription).strip()
        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch, normalizer

    raw_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets["train"].features.keys()).with_format("torch")
    max_input_length = 30.0

    def is_audio_in_length_range(length):
        return length < max_input_length

    raw_datasets["train"] = raw_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    return raw_datasets

def data_collate(raw_datasets, processor):

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

def compute_metrics(pred, processor, normalizer):
    
    metric = evaluate.load("wer")

    do_normalize_eval = True

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def train(model, raw_datasets, data_collator, compute_metrics, processor):
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    #if cuda is available, set fp16 to true
    if torch.cuda.is_available():
        fp16 = True
        
    training_args = Seq2SeqTrainingArguments(
        output_dir="models/whisper-small",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        # max_steps=5000,
        gradient_checkpointing=True,
        fp16=fp16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        # save_steps=1000,
        # eval_steps=1000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        num_train_epochs=1,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    kwargs = {
        "dataset_tags": f"{training_args.dataset_tags}",
        "dataset": f"{training_args.dataset_name}",
        "language": "he",
        "model_name": f"{training_args.model_name_or_path}",
        "finetuned_from": f"{training_args.model_name_or_path}",
        "tasks": "automatic-speech-recognition",
    }
    trainer.save_model(training_args.output_dir)
    # %%
    eval_results = trainer.evaluate()

    print(eval_results)
    
    return trainer
