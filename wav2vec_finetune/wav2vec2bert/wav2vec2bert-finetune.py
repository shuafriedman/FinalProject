import torch
import bitsandbytes as bnb
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor, AutoFeatureExtractor, AutoTokenizer, AutoModel
from transformers import TrainingArguments
from datasets import load_metric, DatasetDict, load_from_disk
from transformers.trainer_pt_utils import get_parameter_names

from config import *
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader
import os

@dataclass
class DataCollatorCTCWithPadding:
    processor: None
    input_key: None
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.input_key == 'input_features':
            input_features = [{"input_features": feature["input_features"]} for feature in features]
        elif self.input_key == 'input_values':
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
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

class SpeechRecognitionModel(pl.LightningModule):
    def __init__(self, model_name, processor, training_args):
        super().__init__()
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            model_name,
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
        self.processor = processor
        self.wer_metric = load_metric("wer")
        self.training_args = training_args

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = outputs.loss
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids)
        labels = batch["labels"]
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels, group_tokens=False)
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('wer', wer, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
            lr=self.training_args.learning_rate,
        )
        return optimizer

class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, processor, input_key, batch_size):
        super().__init__()
        self.processor = processor
        self.input_key = input_key
        self.batch_size = batch_size
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, input_key=self.input_key)

    def prepare_data(self):
        dataset_name = DATASETS[0]['local_path']
        filtered_suffix = "-filtered" if FILTER_LONG_SAMPLES else None
        preprocessing_suffix = "-proc" if PERFORM_PREPROCESSING_ON_DATASET_CREATION else None
        dataset_name = f"{DATA_FOLDER_PATH}/{dataset_name}{filtered_suffix}{preprocessing_suffix}"
        print("loading data from " + dataset_name)
        self.dataset = load_from_disk(dataset_name)
        if DRY_RUN:
            self.dataset = DatasetDict({
                "train": self.dataset["train"].select(range(128)),
                "test": self.dataset["test"].select(range(128))
            })

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, collate_fn=self.data_collator)

def main():
    base_model_path = f"{LOCAL_MODEL_PATH}/{MODEL_CONFIG['model_name']}" if DOWNLOAD_MODEL_LOCALLY else MODEL_CONFIG['model_name']
    model_path = base_model_path + '-finetuned'
    processor = MODEL_CONFIG['processor'].from_pretrained(model_path)
    batch_size = 8 if not DRY_RUN else 2
    training_args = TrainingArguments(
        output_dir='./',
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4 if not DRY_RUN else 2,
        num_train_epochs=1 if not DRY_RUN else 1,
        gradient_checkpointing=True,
        save_steps=600,
        learning_rate=5e-5,
    )

    data_module = SpeechDataModule(processor=processor, input_key=MODEL_CONFIG['input_key'], batch_size=batch_size)
    model = SpeechRecognitionModel(model_name=base_model_path, processor=processor, training_args=training_args)
    
    model_checkpoint = ModelCheckpoint(
        dirpath=LOCAL_MODEL_PATH,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        strategy='auto',
        max_epochs=training_args.num_train_epochs,
        callbacks=[model_checkpoint]
        # gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data_module)
    #save the model

if __name__ == '__main__':
    main()
