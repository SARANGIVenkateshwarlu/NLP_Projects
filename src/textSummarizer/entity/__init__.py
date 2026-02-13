from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path                  #frm config.yaml
    data_path: Path                 #frm config.yaml
    model_ckpt: Path                #frm config.yaml
    num_train_epochs: int           #frm params.yaml
    warmup_steps: int               #frm params.yaml
    per_device_train_batch_size: int #frm params.yaml
    weight_decay: float              #frm params.yaml
    logging_steps: int               #frm params.yaml
    evaluation_strategy: str         #frm params.yaml
    eval_steps: int                  #frm params.yaml
    save_steps: float                 #frm params.yaml
    gradient_accumulation_steps: int  #frm params.yaml