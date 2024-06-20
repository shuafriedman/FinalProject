import torch
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

import datasets
import os
import sys
import pathlib
import subprocess
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor, AutoFeatureExtractor, AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
#import load_metric
from datasets import load_metric, DatasetDict, load_from_disk, load_dataset
from config import *
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import logging
from accelerate import Accelerator, DistributedType
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
        if self.accelerator.distributed_type == DistributedType.XLA:
            max_length = 128
        else:
            max_length = None

        if self.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

    
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

def get_adam8_bit(training_args, model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim

def evaluate(model, dataloader, accelerator, processor, wer_metric):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_items = 0

    # Prepare to collect predictions and references
    predictions = []
    references = []

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
            predictions.extend(batch_predictions)

            # Assuming labels are already IDs, decode them
            # If your labels are in another format, you might need to adjust this
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            batch_references = processor.batch_decode(labels, skip_special_tokens=True)
            references.extend(batch_references)

            tqdm_dataloader.set_description(f"Evaluating (Loss: {total_loss / total_items:.4f})")

    # Compute WER
    wer_score = wer_metric.compute(predictions=predictions, references=references)

    average_loss = total_loss / total_items
    model.train()  # Set the model back to training mode

    # Return both loss and WER
    return average_loss, wer_score

def main():
    model_path =f"{LOCAL_MODEL_PATH}/{MODEL_CONFIG['model_name']}" if DOWNLOAD_MODEL_LOCALLY else MODEL_CONFIG['model_name']
    print("Loading tokenizer")
    print("Loading Processor")
    processor = MODEL_CONFIG['processor'].from_pretrained(model_path + '-finetuned')
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
    print("Checking for cuda")
    print("Getting Training Args")
    batch_size = 4 if not DRY_RUN else 1
    gradient_accumulation = 4 if not DRY_RUN else 2
    num_epochs = 3 if not DRY_RUN else 1
    if train_samples_len:
        max_steps= num_epochs * train_samples_len / batch_size / gradient_accumulation
    else:
        max_steps=128
    training_args = TrainingArguments(
        output_dir= './',
        group_by_length=True,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=True,
        # save_steps=600,
        # max_steps = max_steps,
        # eval_steps=300 if not DRY_RUN else max_steps,
        logging_steps=20,
        learning_rate=5e-5,
        # warmup_steps=500,
        push_to_hub=False,
        optim="adamw_bnb_8bit",
        # torch_compile=True
        # auto_find_batch_size=True
    ) 
    print("Setting Trainer")
# Instead of directly passing 'dataset' to DataLoader, pass dataset['train'] or dataset['test']
    

    optimizer = get_adam8_bit(training_args, model)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    total_steps_per_epoch = train_samples_len // training_args.train_batch_size
    if train_samples_len % training_args.train_batch_size != 0:
        total_steps_per_epoch += 1

    # Calculate the total training steps for all epochs
    total_training_steps = total_steps_per_epoch * training_args.num_train_epochs
    accelerator = Accelerator()

    data_collator = DataCollatorCTCWithPadding(processor=processor, accelerator=accelerator, input_key=MODEL_CONFIG['input_key'], padding=True)
    wer_metric = load_metric("wer", trust_remote_code=True)
    compute_metric= compute_metrics_custom(wer_metric, processor)

    if USE_TRAINER:
        model = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metric,
        train_dataset= dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        # optimizers=(optimizer, None)
        )
        print("Beginning Training")
        model.train()
        model.save_model(f"{model_path}-finetuned")

    else:
        dataloaders = {
            'train': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator),
            'test': DataLoader(dataset['test'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
        }

        
        dataloaders['train'], dataloaders['test'], model, optimizer = accelerator.prepare(
            dataloaders['train'],  dataloaders['test'], model, optimizer
        )
        progress_bar = tqdm(range(train_samples_len // training_args.train_batch_size), desc="Training")

        for epoch in range(training_args.num_train_epochs):
            model.train()
            # Loop over the number of epochs
            for step, batch in enumerate(dataloaders['train'], start=1):
                with accelerator.accumulate(model):
                    for key in batch:
                        if batch[key].dtype == torch.float16:
                            batch[key] = batch[key].to(torch.float32)
                    loss = model(**batch).loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    # This condition is adjusted to reflect the current position within the epoch
                    current_global_step = step + epoch * total_steps_per_epoch
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch + 1}/{training_args.num_train_epochs}")

                    # Evaluate at the end of each epoch
                    if current_global_step % total_steps_per_epoch == 0 or current_global_step == total_training_steps:
                        eval_loss, wer = evaluate(model, dataloaders['test'], accelerator, processor, wer_metric)
                        logger.info(f"Global Step: {current_global_step}, Epoch: {epoch + 1}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}, WER: {wer}")

            progress_bar.reset(total=total_steps_per_epoch)  # Reset for the next epoch if you are using a single progress bar for all epochs

        progress_bar.close()
        print("Finished Training")
        model.save_pretrained(f"{model_path}-finetuned")
if __name__=='__main__':
   main()