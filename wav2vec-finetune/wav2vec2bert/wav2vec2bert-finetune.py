import torch
import datasets
import os
import sys
import pathlib
import subprocess
from transformers import Wav2Vec2CTCTokenizer
from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor
from transformers import Trainer, TrainingArguments
#import load_metric
from datasets import load_metric, DatasetDict
from config import *
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
    
def compute_metrics_custom(wer_metric, processor):
    def compute_metric(pred, ):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metric

def load_dataset_from_disk():
    dataset = datasets.load_from_disk(f"{DATA_FOLDER_PATH}/fleurs")
    if DRY_RUN:
        small_train_subset = dataset['train'].select(range(32))
        small_test_subset = dataset['test'].select(range(32)) 
        dataset = DatasetDict({"train": small_train_subset, "test": small_test_subset})
    return dataset

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

def main():
    print("Loading tokenizer")
    print("Loading Processor")
    processor = Wav2Vec2BertProcessor.from_pretrained(LOCAL_MODEL_PATH if DOWNLOAD_MODEL_LOCALLY else None) #TODO
    print("Loading Dataset")
    dataset = load_dataset_from_disk()
    print("Mapping Dataset Processor to Dataset")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].features.keys(), \
                        fn_kwargs={"processor": processor})
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")
    compute_metric= compute_metrics_custom(wer_metric, processor)
    print("Defining Model and Arguements")
    path = LOCAL_MODEL_PATH if DOWNLOAD_MODEL_LOCALLY else BASE_MODEL_NAME
    model = Wav2Vec2BertForCTC.from_pretrained(
        path,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    print("Checking for cuda")
    fp16= True if torch.cuda.is_available() else False
    print("Getting Training Args")
    training_args = TrainingArguments(
        output_dir= './',
        group_by_length=True,
        per_device_train_batch_size= 16 if not DRY_RUN else 1,
        gradient_accumulation_steps=2 if not DRY_RUN else 5,
        evaluation_strategy="steps",
        num_train_epochs=1 if not DRY_RUN else 1,
        gradient_checkpointing=True,
        fp16=fp16,
        save_steps=600,
        eval_steps=300 if not DRY_RUN else 8,
        logging_steps=300,
        learning_rate=5e-5,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )
    print("Setting Trainer")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metric,
        train_dataset= dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )
    print("Beginning Training")
    trainer.train()
    print("Saving Model")
    trainer.save_model(FINETUNED_MODEL_PATH)
if __name__=='__main__':
   main()

