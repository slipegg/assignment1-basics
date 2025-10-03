from dataclasses import dataclass

@dataclass
class TrainingConfig():
    batch_size: int
    context_length: int

    device: str = "cpu"

    gradient_clipping_max_norm: float 

    # training
    total_steps: int
    eval_interval: int

    # evaluation
    eval_batches: int

    # checkpointing
    checkpoint_folder: str
    checkpoint_interval: int
    checkpoint_prefix: str = "checkpoint_"   
    
    # dataset
    train_dataset_path: str
    valid_dataset_path: str

    # model
    vocab_size: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float = 100000.0
    dtype: str = "float32"  # "float32", "bfloat16", "float16"

    # optimizer
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2

    # learning_rate_schedule
    warmup_steps: int = 100
    cosine_steps: int = 10000
    min_learning_rate: float = 1e-4

    # wandb
    project_name: str = "cs336-assignment1"
    run_name: str | None = None
    log_interval: int = 10