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
from datasets import load_metric, DatasetDict, load_from_disk, load_dataset
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
    dataset_name = DATASETS[0]['local_path']
    suffix = "-filtered" if FILTER_LONG_SAMPLES else None
    if FROM_HUB:
        dataset=load_dataset(f"Shuaf98/{dataset_name}{suffix}", streaming=True)
        train_samples_len=None
    else:
        dataset= load_from_disk(f"{DATA_FOLDER_PATH}/{dataset_name}{suffix}")
        train_samples_len=dataset['train'].num_rows
        for name in dataset.keys():
            dataset[name] = dataset[name].to_iterable_dataset()
    if DRY_RUN:
        small_train_subset = dataset['train'].select(range(128))
        small_test_subset = dataset['test'].select(range(128)) 
        dataset = DatasetDict({"train": small_train_subset, "test": small_test_subset})
    return dataset, train_samples_len

def main():
    print("Loading tokenizer")
    print("Loading Processor")
    processor = Wav2Vec2BertProcessor.from_pretrained(LOCAL_MODEL_PATH if DOWNLOAD_MODEL_LOCALLY else None) #TODO
    print("Loading Dataset")
    dataset, train_samples_len = load_dataset_from_disk()
    print("Mapping Dataset Processor to Dataset")

    
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
        vocab_size=len(processor.tokenizer),
    )
    print("Checking for cuda")
    fp16= True if torch.cuda.is_available() else False
    print("Getting Training Args")
    batch_size = 4 if not DRY_RUN else 1
    gradient_accumulation = 8 if not DRY_RUN else 2
    num_epochs = 1 if not DRY_RUN else 1
    if train_samples_len:
        max_steps= num_epochs * train_samples_len / batch_size / gradient_accumulation
    else:
        max_steps=None
    training_args = TrainingArguments(
        output_dir= './',
        # group_by_length=True,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy="steps",
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=fp16,
        save_steps=600,
        max_steps = max_steps,
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
# from accelerate import Accelerator
# from torch.utils.data.dataloader import DataLoader

# dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

# if training_args.gradient_checkpointing:
#     model.gradient_checkpointing_enable()

# accelerator = Accelerator(fp16=training_args.fp16)
# model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

# model.train()
# for step, batch in enumerate(dataloader, start=1):
#     loss = model(**batch).loss
#     loss = loss / training_args.gradient_accumulation_steps
#     accelerator.backward(loss)
#     if step % training_args.gradient_accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()
if __name__=='__main__':
   main()

