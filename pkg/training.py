import os
import time
import typing
import torch
import wandb
import tqdm
import numpy as np
import numpy.typing as npt

from training_config import TrainingConfig
from optimizer import cross_entropy, learning_rate_schedule, gradient_clipping, AdamW
from model import TransformerLM


def data_loading(
        x: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
    )-> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    all_indices = start_indices[:, None] + np.arange(context_length + 1)[None, :]

    inputs = x[all_indices[:, :-1]]
    targets = x[all_indices[:, 1:]]

    return tuple([torch.from_numpy(t).to(device) for t in (inputs, targets)])


def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],  
        model: torch.nn.Module,  
        optimizer: torch.optim.Optimizer
    ) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def training_step(
        config: TrainingConfig,
        dataset: npt.NDArray,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
    input, target = data_loading(x=dataset, batch_size=config.batch_size, context_length=config.context_length, device=config.device)

    model.train()
    optimizer.zero_grad()

    logits = model(input)
    loss = cross_entropy(logits, target)
    loss.backward()
    gradient_clipping(model.parameters(), max_norm=config.gradient_clipping_max_norm)

    optimizer.step()

    return loss.item()

def evaluate(
        config: TrainingConfig,
        dataset: npt.NDArray,
        model: torch.nn.Module
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(config.eval_batches):
            input, target = data_loading(x=dataset, batch_size=config.batch_size, context_length=config.context_length, device=config.device)
            logits = model(input)
            loss = cross_entropy(logits, target)
            total_loss += loss.item()

    return total_loss / config.eval_batches

def training(
        config: TrainingConfig
    ):
    wandb.init(project=config.project_name, 
               name=config.run_name if config.run_name is not None else "run-"+ time.strftime("%Y%m%d-%H%M%S"),
               config=config.__dict__,
    )
    print("WandB initialized, project name:", config.project_name, "run name:", wandb.run.name)

    train_dataset = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    valid_dataset = np.memmap(config.valid_path, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=config.dtype
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas = config.betas,
        eps = config.eps,
        weight_decay=config.weight_decay
    )

    start_step = 1
    if config.checkpoint_folder is not None:
        if os.path.isdir(config.checkpoint_folder) == False or not os.path.exists(config.checkpoint_folder):
            os.makedirs(config.checkpoint_folder)
            print(f"Created checkpoint folder at {config.checkpoint_folder}")
        else:
            checkpoint_files = [f for f in os.listdir(config.checkpoint_folder) if f.startswith(config.checkpoint_prefix) and f.endswith(".pt")]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split(config.checkpoint_prefix)[1].split(".pt")[0]))
                checkpoint_path = os.path.join(config.checkpoint_folder, latest_checkpoint)
                start_step = load_checkpoint(checkpoint_path, model, optimizer) + 1
                print(f"Resumed from checkpoint {checkpoint_path}, starting at step {start_step}")
    
    start_time = time.time()
    for step in tqdm(range(start_step, config.total_steps+1)):
        lr = learning_rate_schedule(step, warmup_steps=config.warmup_steps, max_lr=config.learning_rate, min_lr=config.min_learning_rate, cosine_steps=config.cosine_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = training_step(config, train_dataset, model, optimizer)

        if (step - 1) % config.log_interval == 0 or step == config.total_steps:
            grad_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in model.parameters() if p.grad is not None)).item()
            wandb.log(
                {
                    "step": step,
                    "train/loss": train_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "train/wallclock_time": time.time() - start_time
                }
            )
        
        if (step-1) % config.eval_interval == 0 or step == config.total_steps:
            valid_loss = evaluate(config, valid_dataset, model)
            wandb.log(
                {
                    "step": step,
                    "valid/loss": valid_loss,
                    "valid/wallclock_time": time.time() - start_time
                }
            )
            print(f"Step {step}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")
        
        if ((step-1) % config.checkpoint_interval == 0 and step > 1) or step == config.total_steps:
            save_path = f"{config.checkpoint_prefix}{step}.pt"
            save_checkpoint(model, optimizer, step, save_path)
            print(f"Checkpoint saved to {save_path}")
    
    wandb.finish()
    print("Training completed.")
