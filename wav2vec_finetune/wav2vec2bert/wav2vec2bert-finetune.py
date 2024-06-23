import torch
import bitsandbytes as bnb
from torch.optim import AdamW
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
import math
import datasets
import os
import sys
import pathlib
import subprocess
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor, AutoFeatureExtractor, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
#import load_metric
from datasets import load_metric, DatasetDict, load_from_disk, load_dataset
from config import *
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator, DistributedType
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import logging
from typing import List, Dict, Union
import torch

@dataclass
class DataCollatorCTCWithPadding:
    def __init__(self, processor, accelerator, input_key='input_features', padding=True):
        self.processor = processor
        self.accelerator= accelerator
        self.input_key = input_key
        self.padding = padding

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.input_key == 'input_features':
            input_features = [{"input_features": feature["input_features"]} for feature in features]
        elif self.input_key == 'input_values':
            input_features = [{"input_values": feature["input_values"]} for feature in features]

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Determine pad_to_multiple_of based on accelerator's mixed precision
        # if self.accelerator.distributed_type == DistributedType.XLA:
        #     max_length = 128
        # else:
        #     max_length = None

        if self.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            # max_length=max_length,
            # pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            # max_length=max_length,
            # pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

def get_adam8_bit(adam_args, model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": adam_args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": adam_args["weight_decay"],
        },
    ]

    optimizer_kwargs = {
        "betas": (adam_args["adam_beta1"], adam_args["adam_beta2"]),
        "eps": adam_args["adam_epsilon"],
    }
    optimizer_kwargs["lr"] = adam_args["learning_rate"]
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(adam_args["adam_beta1"], adam_args["adam_beta2"]),
        eps=adam_args["adam_epsilon"],
        lr=adam_args["learning_rate"],
    )
    return adam_bnb_optim

def compute_metrics_custom(wer_metric, processor):
    def compute_metric(pred, ):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metric

def load_dataset_from_disk():
    dataset_name = DATASETS[0]['local_path']
    filtered_suffix = "-filtered" if FILTER_LONG_SAMPLES else None
    preprocessing_suffix="-proc" if PERFORM_PREPROCESSING_ON_DATASET_CREATION else None
    # if FROM_HUB:
    #     dataset=load_dataset(f"Shuaf98/{dataset_name}{suffix}", streaming=True)
    #     train_samples_len=None
    # else:
    #     dataset= load_from_disk(f"{DATA_FOLDER_PATH}/{dataset_name}{suffix}")
    #     train_samples_len=dataset['train'].num_rows
    #     for name in dataset.keys():
    #         dataset[name] = dataset[name].to_iterable_dataset()
    dataset_name=f"{DATA_FOLDER_PATH}/{dataset_name}{filtered_suffix}{preprocessing_suffix}"
    print("Loading Dataset : ", dataset_name)
    dataset= load_from_disk(dataset_name)
    train_samples_len=dataset['train'].num_rows
    if DRY_RUN:
        print("creating dataset for dry run")
        small_train_subset = dataset['train'].select(range(128))
        train_samples_len = 128
        small_test_subset = dataset['test'].select(range(128)) 
        dataset = DatasetDict({"train": small_train_subset, "test": small_test_subset})

    return dataset, train_samples_len

def evaluate(model, dataloader, accelerator, processor, wer_metric):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_items = 0

    # Prepare to collect predictions and references

    with torch.no_grad(), tqdm(dataloader, desc="Evaluating", leave=False) as tqdm_dataloader:
        for batch in tqdm_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += accelerator.gather(loss).item() * batch[MODEL_CONFIG['input_key']].size(0)
            total_items += batch[MODEL_CONFIG['input_key']].size(0)

            # Decode the predicted IDs to text
            
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch_predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
            # Assuming labels are already IDs, decode them
            # If your labels are in another format, you might need to adjust this
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            batch_references = processor.batch_decode(labels, skip_special_tokens=True)
            wer_metric.add_batch(
                predictions=batch_predictions,
                references=batch_references
            )
            if len(batch_predictions) > 0 and len(batch_references) > 0:
                print("Sample Prediction:")
                print(batch_predictions[0])
                print("Sample Reference:")
                print(batch_references[0])
            tqdm_dataloader.set_description(f"Evaluating (Loss: {total_loss}")

    # Compute WER
    wer_score = wer = 100 * wer_metric.compute()

    average_loss = total_loss / total_items
    model.train()  # Set the model back to training mode

    # Return both loss and WER
    return average_loss, wer_score

def main():
    batch_size = 16 if not DRY_RUN else 1
    gradient_accumulation_steps = 2 if not DRY_RUN else 1
    num_train_epochs = 2 if not DRY_RUN else 1
    learning_rate=1e-5
    max_train_steps=None
    adam_args= {"adam_beta1": 0.9, "adam_beta2": 0.999,  "adam_epsilon": 1e-8, "weight_decay" : 0.0, "learning_rate": learning_rate}
    
    accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=gradient_accumulation_steps)

    model_path =f"{LOCAL_MODEL_PATH}/{MODEL_CONFIG['model_name']}" if DOWNLOAD_MODEL_LOCALLY else MODEL_CONFIG['model_name']
    print("Loading tokenizer")
    print("Loading Processor")
    processor = MODEL_CONFIG['processor'].from_pretrained(model_path + '-finetuned2')
    print("Loading Dataset")
    dataset, train_samples_len = load_dataset_from_disk()
    print("Mapping Dataset Processor to Dataset")

    print("Defining Model and Arguements")
    if MANUALLY_SET_MODEL_CONFIG:
        model = MODEL_CONFIG['model'].from_pretrained(
            model_path,
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
    else:
        model=MODEL_CONFIG['model'].from_pretrained(
            model_path,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )

    max_steps= num_train_epochs * train_samples_len / batch_size / gradient_accumulation_steps

    print("Setting Trainer")
    
    optimizer = get_adam8_bit(adam_args, model)
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.995)) # add bnb optimizer
    # optimizer = AdamW(model.parameters(), lr=learning_rate)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Calculate the total training steps for all epochs
    data_collator = DataCollatorCTCWithPadding(processor=processor, accelerator=accelerator, input_key=MODEL_CONFIG['input_key'], padding=True)
    wer_metric = load_metric("wer", trust_remote_code=True)
    compute_metric= compute_metrics_custom(wer_metric, processor)

    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, collate_fn=data_collator),
        'test': DataLoader(dataset['test'], batch_size=batch_size, collate_fn=data_collator)
    }

    num_update_steps_per_epoch = math.ceil(len(dataloaders["train"]) / gradient_accumulation_steps)

    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

    model.gradient_checkpointing_enable()
    for param in model.wav2vec2_bert.encoder.parameters():
        param.requires_grad = False
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100, #20
        num_training_steps=max_train_steps
    )
    dataloaders['train'], dataloaders['test'], model, optimizer, lr_scheduler = accelerator.prepare(
        dataloaders['train'],  dataloaders['test'], model, optimizer, lr_scheduler
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_samples_len}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        # Loop over the number of epochs
        total_loss= 0
        for step, batch in enumerate(dataloaders['train'], start=1):
            loss = model(**batch).loss
            
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss}", total_loss = f"{total_loss}", epoch=f"{epoch + 1}/{num_train_epochs}")

            if step % 20==0:
                eval_loss, wer = evaluate(model, dataloaders['test'], accelerator, processor, wer_metric)
                logger.info(f"Epoch: {epoch + 1}, Training Loss: {total_loss / train_samples_len}, Evaluation Loss: {eval_loss}, WER: {wer}")

        eval_loss, wer = evaluate(model, dataloaders['test'], accelerator, processor, wer_metric)
        logger.info(f"Epoch: {epoch + 1}, Training Loss: {total_loss / train_samples_len}, Evaluation Loss: {eval_loss}, WER: {wer}")

        progress_bar.reset(total=max_train_steps)  # Reset for the next epoch if you are using a single progress bar for all epochs
    
    progress_bar.close()
    print("Finished Training")
    model.save_pretrained(f"{model_path}-finetuned")
if __name__=='__main__':
   main()