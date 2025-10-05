import torch
import os
from cs336_basics.models.model import TransformerLM
from cs336_basics.config.training_config import TrainingConfig
from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.utils.generating import generate

def load_model_from_checkpoint(config: TrainingConfig) -> TransformerLM:
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
    
    if config.is_compile:
        model = torch.compile(model, backend="eager")
        print("Model compiled with torch.compile")
    
    if config.checkpoint_folder is not None:
        if os.path.isdir(config.checkpoint_folder) == False or not os.path.exists(config.checkpoint_folder):
            print(f"Checkpoint folder {config.checkpoint_folder} does not exist.")
        else:
            checkpoint_files = [f for f in os.listdir(config.checkpoint_folder) if f.startswith(config.checkpoint_prefix) and f.endswith(".pt")]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split(config.checkpoint_prefix)[1].split(".pt")[0]))
                checkpoint_path = os.path.join(config.checkpoint_folder, latest_checkpoint)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Resumed from checkpoint {checkpoint_path}")
    
    return model

if __name__ == "__main__":
    config = TrainingConfig(
        project_name="cs336-tinystories",
        total_steps=5000,
        checkpoint_interval=50,
    )
    model = load_model_from_checkpoint(config)
    prompt = "Once upon a time, there was a little girl named Tomato. Tomato "
    prompt = "Once upon a time, there was a pretty girl named Lily. She"
    tokenizer = Tokenizer.from_files(vocab_filepath = config.vocab_path, 
                                merges_filepath = config.merge_path, 
                                special_tokens=config.special_tokens)
    prompt_token = tokenizer.encode(prompt)
    end_token_id = tokenizer.encode('<|endoftext|>')[0]
    # print("Prompt token ids:\n", prompt_token)
    # print("End token id:\n", end_token_id)
    generate_token = generate(
        model=model,
        input_ids=torch.tensor(prompt_token, dtype=torch.long),
        max_seq_len=config.context_length,
        max_generate_length=config.context_length*2,
        temperature=0,
        top_k=10,
        end_token_id=end_token_id,
        device=config.device,
        tokenizer=tokenizer,      # ✅ 传入 tokenizer
        stream_output=True        # ✅ 实时输出
    )
    generate_text = tokenizer.decode(generate_token)
    # print("Generated text:\n", generate_text)

    