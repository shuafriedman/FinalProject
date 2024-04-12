import torch
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

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
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import logging
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
    filtered_suffix = "-filtered" if FILTER_LONG_SAMPLES else None
    preprocessing_suffix="-prox" if PERFORM_PREPROCESSING_ON_DATASET_CREATION else None
    # if FROM_HUB:
    #     dataset=load_dataset(f"Shuaf98/{dataset_name}{suffix}", streaming=True)
    #     train_samples_len=None
    # else:
    #     dataset= load_from_disk(f"{DATA_FOLDER_PATH}/{dataset_name}{suffix}")
    #     train_samples_len=dataset['train'].num_rows
    #     for name in dataset.keys():
    #         dataset[name] = dataset[name].to_iterable_dataset()
    dataset= load_from_disk(f"{DATA_FOLDER_PATH}/{dataset_name}{filtered_suffix}{preprocessing_suffix}")
    train_samples_len=dataset['train'].num_rows
    if DRY_RUN:
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
            total_loss += accelerator.gather(loss).item() * batch["input_features"].size(0)
            total_items += batch["input_features"].size(0)

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
    if MANUALLY_SET_MODEL_CONFIG:
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
    else:
        model=Wav2Vec2BertForCTC.from_pretrained(
            path,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )
    print("Checking for cuda")
    fp16= "True" if torch.cuda.is_available() else False
    print("Getting Training Args")
    batch_size = 8 if not DRY_RUN else 1
    gradient_accumulation = 4 if not DRY_RUN else 2
    num_epochs = 6 if not DRY_RUN else 1
    if train_samples_len:
        max_steps= num_epochs * train_samples_len / batch_size / gradient_accumulation
    else:
        max_steps=None
    training_args = TrainingArguments(
        output_dir= './',
        group_by_length=True,
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
        optim="adamw_bnb_8bit"
    )
    print("Setting Trainer")
    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=compute_metric,
    #     train_dataset= dataset["train"],
    #     eval_dataset=dataset["test"],
    #     tokenizer=processor.feature_extractor,
    # )
    # print("Beginning Training")
    # trainer.train()
    # print("Saving Model")
    # trainer.save_model(FINETUNED_MODEL_PATH)

# Instead of directly passing 'dataset' to DataLoader, pass dataset['train'] or dataset['test']
    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count()),
        'test': DataLoader(dataset['test'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count())
    }
    progress_bar = tqdm(range(train_samples_len // training_args.train_batch_size), desc="Training")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = get_adam8_bit(training_args, model)
    accelerator = Accelerator(mixed_precision= "fp16" if torch.cuda.is_available() else "no")
    dataloaders['train'], dataloaders['test'], model, optimizer = accelerator.prepare(
         dataloaders['train'],  dataloaders['test'], model, optimizer
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    total_steps_per_epoch = train_samples_len // training_args.train_batch_size
    if train_samples_len % training_args.train_batch_size != 0:
        total_steps_per_epoch += 1

    # Calculate the total training steps for all epochs
    total_training_steps = total_steps_per_epoch * training_args.num_train_epochs

    for epoch in range(training_args.num_train_epochs):
        model.train()
          # Loop over the number of epochs
        for step, batch in enumerate(dataloaders['train'], start=1):
            loss = model(**batch).loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == total_steps_per_epoch:
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
    model.save_pretrained(FINETUNED_MODEL_PATH)
if __name__=='__main__':
   main()

