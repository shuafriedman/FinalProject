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
from transformers import Trainer, get_linear_schedule_with_warmup
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
    processor: None
    input_key: None
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        if self.input_key=='input_features':
            input_features = [{"input_features": feature["input_features"]} for feature in features]
        elif self.input_key=='input_values':
            input_features = [{"input_values": feature["input_values"]} for feature in features]

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
class CustomTrainingArguements:
    def __init__(self, output_dir, group_by_length, per_device_train_batch_size, gradient_accumulation_steps, evaluation_strategy, num_train_epochs, gradient_checkpointing, fp16, logging_steps, learning_rate, max_train_steps):
        self.output_dir = output_dir
        self.group_by_length = group_by_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.evaluation_strategy = evaluation_strategy
        self.num_train_epochs = num_train_epochs
        self.gradient_checkpointing = gradient_checkpointing
        self.fp16 = fp16
        self.logging_steps = logging_steps
        self.learning_rate = learning_rate
        self.max_train_steps=max_train_steps
    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
    
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'CustomTrainingArguments' object has no attribute '{name}'")
    
    def __delattr__(self, name: str) -> None:
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            raise AttributeError(f"'CustomTrainingArguments' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value
    
    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

        
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

    
    data_collator = DataCollatorCTCWithPadding(processor=processor, input_key=MODEL_CONFIG['input_key'], padding=True)
    wer_metric = load_metric("wer")
    compute_metric= compute_metrics_custom(wer_metric, processor)
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
            path,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )
    print("Checking for cuda")
    fp16= "True" if torch.cuda.is_available() else False
    print("Getting Training Args")
    batch_size = 4 if not DRY_RUN else 1
    gradient_accumulation = 4 if not DRY_RUN else 2
    num_epochs = 3 if not DRY_RUN else 1
    if train_samples_len:
        max_steps= num_epochs * train_samples_len / batch_size / gradient_accumulation
    else:
        max_steps=128
    
    training_args = CustomTrainingArguements(
        output_dir= './',
        group_by_length=True,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=fp16,
        # save_steps=600,
        # max_steps = max_steps,
        # eval_steps=300 if not DRY_RUN else max_steps,
        logging_steps=20,
        learning_rate=5e-5,
        # warmup_steps=500,
        max_train_steps= None
        # torch_compile=True
        # auto_find_batch_size=True
    ) 
    print("Setting Trainer")

    
    adam_args= {"adam_beta1": 0.9, "adam_beta2": 0.999,  "adam_epsilon": 1e-8, "weight_decay" : 0.0, "learning_rate": training_args.learning_rate}
    optimizer = get_adam8_bit(adam_args, model)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
            'train': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count()),
            'test': DataLoader(dataset['test'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count())
        }

        accelerator = Accelerator(mixed_precision= "fp16" if torch.cuda.is_available() else "no")

        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes
        total_steps_per_epoch = train_samples_len // total_batch_size
        if train_samples_len % total_batch_size != 0:
            total_steps_per_epoch += 1
        total_training_steps = total_steps_per_epoch * training_args.num_train_epochs

        if training_args.max_train_steps is None:
            max_train_steps = training_args.num_train_epochs * total_training_steps

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100, #20
            num_training_steps=max_train_steps
        )
        dataloaders['train'], dataloaders['test'], model, optimizer = accelerator.prepare(
            dataloaders['train'],  dataloaders['test'], model, optimizer
        )
        progress_bar = tqdm(range(total_steps_per_epoch), desc="Training")
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        for epoch in range(training_args.num_train_epochs):
            model.train()
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
        model.save_pretrained(f"{model_path}-finetuned")
if __name__=='__main__':
   main()

