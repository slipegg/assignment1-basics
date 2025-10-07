import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig():
    # wandb
    enable_wandb: bool = True
    project_name: str = "cs336-tinystories"
    run_name: str | None = None
    log_interval: int = 10

    # bpe
    vocab_size: int = 10000
    train_bpe_file: str = "data/TinyStoriesV2-GPT4-train.txt"
    use_pretoken_file: bool = True
    pretoken_path: str = "data/TinyStoriesV2-pretokens.pkl"
    vocab_path: str = "data/TinyStoriesV2-vocab.pkl"
    merge_path: str = "data/TinyStoriesV2-merges.pkl"
    special_tokens: tuple = ('<|endoftext|>',)

    # dataset
    train_dataset_original_path: str = "data/TinyStoriesV2-GPT4-train.txt"
    valid_dataset_original_path: str = "data/TinyStoriesV2-GPT4-valid.txt"
    train_dataset_path: str = "data/TinyStoriesV2-GPT4-train.npy"
    valid_dataset_path: str = "data/TinyStoriesV2-GPT4-valid.npy"

    # training
    total_steps: int = 3000
    batch_size: int = 64
    context_length: int = 256
    gradient_clipping_max_norm: float  = 1.0
    is_compile: bool = True  # use torch.compile or not

    # evaluation
    eval_batches: int = 64
    eval_interval: int = 100

    # checkpointing
    checkpoint_folder: str = "./checkpoints/"
    checkpoint_prefix: str = "checkpoint_"   
    checkpoint_interval: int = 500

    # model
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 100000.0
    dtype = torch.float32

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optimizer
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2

    # learning_rate_schedule
    warmup_steps: int = 100
    cosine_steps: int = 3000
    min_learning_rate: float = 1e-4
