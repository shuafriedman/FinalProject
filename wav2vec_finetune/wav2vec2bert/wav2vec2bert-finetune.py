import torch
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
import itertools

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
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import logging

from utils import CustomTrainingArguements, DataCollatorCTCWithPadding

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
            with torch.cuda.amp.autocast():

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

def log_gradients(model, when):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        else:
            print(f"{when} No gradient for {name}")
    total_norm = total_norm ** 0.5
    print(f"{when} Total grad norm: {total_norm:.4f}")


def main():
    batch_size = 8 if not DRY_RUN else 1
    gradient_accumulation = 2 if not DRY_RUN else 2
    num_epochs = 2 if not DRY_RUN else 1
    if hasattr(CHECKPOINTING_STEPS, "isdigit"):
        if CHECKPOINTING_STEPS == "epoch":
            checkpointing_steps = CHECKPOINTING_STEPS
        elif CHECKPOINTING_STEPS.isdigit():
            checkpointing_steps = int(CHECKPOINTING_STEPS)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{CHECKPOINTING_STEPS}` passed."
            )
    else:
        checkpointing_steps = None
        
    model_path =f"{LOCAL_MODEL_PATH}/{MODEL_CONFIG['model_name']}" if DOWNLOAD_MODEL_LOCALLY else MODEL_CONFIG['model_name']
    print("Loading tokenizer")
    print("Loading Processor")
    processor = MODEL_CONFIG['processor'].from_pretrained(model_path + '-finetuned')
    print("Loading Dataset")
    dataset, train_samples_len = load_dataset_from_disk()
    print("Mapping Dataset Processor to Dataset")

    
    data_collator = DataCollatorCTCWithPadding(processor=processor, input_key=MODEL_CONFIG['input_key'], padding=True)
    wer_metric = load_metric("wer")
    print("Defining Model and Arguements")
    if RESUME_FROM_CHECKPOINT ==False or os.path.exists(f"{model_path}-finetuned/checkpoints"):
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
            f"{model_path}-finetuned",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )
    model.config.ctc_zero_infinity = True #necessary, data keeps giving inf in loss towards end of first epoch
    print("Getting Training Args")

    # if train_samples_len:
    #     max_steps= num_epochs * train_samples_len / batch_size / gradient_accumulation
    max_steps=None
    training_args = CustomTrainingArguements(
        output_dir=f"{model_path}-finetuned",
        group_by_length=True,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        # save_steps=600,
        max_steps = max_steps,
        # eval_steps=300 if not DRY_RUN else max_steps,
        logging_steps=50,
        learning_rate=5e-5,
        # warmup_steps=500,
        # torch_compile=True
        # auto_find_batch_size=True
    ) 
    print("Setting Trainer")
# Instead of directly passing 'dataset' to DataLoader, pass dataset['train'] or dataset['test']
    
    adam_args= {"adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8, "weight_decay" : 0.00, "learning_rate": training_args.learning_rate}
    optimizer = get_adam8_bit(adam_args, model)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count()),
        'test': DataLoader(dataset['test'], batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, num_workers=os.cpu_count())
    }
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    
    training_args.train_batch_size = training_args.per_device_train_batch_size * max(1, accelerator.num_processes) #from huggingface trainer args code
    total_steps_per_epoch = train_samples_len // training_args.train_batch_size
    if train_samples_len % training_args.train_batch_size != 0:
        total_steps_per_epoch += 1

    total_training_steps = total_steps_per_epoch * training_args.num_train_epochs

    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=total_training_steps * 0.05,
    #     num_training_steps=total_training_steps
    # )
    dataloaders['train'], dataloaders['test'], model, optimizer = accelerator.prepare(
        dataloaders['train'],  dataloaders['test'], model, optimizer
    )
    resume_step = None
    overall_step = 0
    starting_epoch = 0
    if RESUME_FROM_CHECKPOINT:
        if RESUME_FROM_SPECIFIC_CHECKPOINT:
            checkpoint_path = f"{model_path}-finetuned/{RESUME_FROM_CHECKPOINT_DIR}/{RESUME_FROM_SPECIFIC_CHECKPOINT}"
            accelerator.print(f"Resumed from checkpoint: {RESUME_FROM_SPECIFIC_CHECKPOINT}")
            accelerator.load_state(checkpoint_path)
            path = os.path.basename(checkpoint_path)
        else:
            checkpoint_path = f"{model_path}-finetuned/{RESUME_FROM_CHECKPOINT_DIR}"
    # We also need to keep track of the stating epoch so files are named properly
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(checkpoint_path) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            latest_checkpoint_dir = dirs[-1]  # Most recent checkpoint is the last
            latest_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint_dir)
            accelerator.print(f"Resumed from latest checkpoint: {latest_checkpoint_path}")
            accelerator.load_state(latest_checkpoint_path)
            path = os.path.basename(latest_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(dataloaders['train'])
            resume_step -= starting_epoch * len(dataloaders['train'])

    # Initialize the progress bar correctly based on resume point
    if resume_step is not None:
        # Calculate total steps per epoch assuming `train_dataloader` is already defined
        total_steps_per_epoch = len(dataloaders['train'])
        initial_step = resume_step
    else:
        total_steps_per_epoch = len(dataloaders['train'])
        initial_step = 0

    progress_bar = tqdm(range(1, total_steps_per_epoch + 1), desc="Training", initial=initial_step)
        
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        if RESUME_FROM_CHECKPOINT and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(dataloaders["train"], resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = dataloaders["train"]
        for step, batch in enumerate(active_dataloader, start=1):
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                logger.info(loss)
                loss = loss / training_args.gradient_accumulation_steps
                logger.info(loss)
                accelerator.backward(loss)
                # log_gradients(model, "Before Clipping")

                if step % training_args.gradient_accumulation_steps == 0 or step == total_steps_per_epoch:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # log_gradients(model, "After Clipping")
                    optimizer.step()
                    optimizer.zero_grad()
                    
                overall_step += 1
                if isinstance(checkpointing_steps, int):
                    output_dir = f"{checkpoint_path}/step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        if training_args.output_dir is not None:
                            output_dir = os.path.join(training_args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch + 1}/{training_args.num_train_epochs}")
            # Evaluate at the end of each epoch
            current_global_step = step + epoch * total_steps_per_epoch

            if current_global_step % training_args.logging_steps == 0:
                eval_loss, wer = evaluate(model, dataloaders['test'], accelerator, processor, wer_metric)
                logger.info(f"Global Step: {current_global_step}, Epoch: {epoch + 1}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}, WER: {wer}")
        #evaluate if logging steps is None or if we are at the end of the last epoch
        # if training_args.logging_steps == None or (epoch + 1 == training_args.num_train_epochs):    
        eval_loss, wer = evaluate(model, dataloaders['test'], accelerator, processor, wer_metric)
        logger.info(f"Global Step: {current_global_step}, Epoch: {epoch + 1}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}, WER: {wer}")
        
        if checkpointing_steps == "epoch":
            output_dir = f"{checkpoint_path}/epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        progress_bar.reset(total=total_steps_per_epoch)  # Reset for the next epoch if you are using a single progress bar for all epochs

    progress_bar.close()
    print("Finished Training")
    model.save_pretrained(f"{model_path}-finetuned")
if __name__=='__main__':
   main()

