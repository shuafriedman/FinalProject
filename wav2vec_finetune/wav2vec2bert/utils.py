from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

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
    def __init__(self, output_dir, group_by_length, per_device_train_batch_size, gradient_accumulation_steps, evaluation_strategy, num_train_epochs, gradient_checkpointing, logging_steps, learning_rate, max_steps):
        self.output_dir = output_dir
        self.group_by_length = group_by_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.evaluation_strategy = evaluation_strategy
        self.num_train_epochs = num_train_epochs
        self.gradient_checkpointing = gradient_checkpointing
        self.logging_steps = logging_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps

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