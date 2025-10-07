
import os
import pathlib

from cs336_basics.config.training_config import TrainingConfig
from cs336_basics.bpe.train_bpe import train_bpe
from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.training.trainer import training

def init_dataset(config: TrainingConfig):
    if not os.path.exists(config.vocab_path) or not os.path.exists(config.merge_path):
        train_bpe(input_path = config.train_path,
                vocab_size = config.vocab_size,
                special_tokens = config.special_tokens,
                use_pretoken_file = config.use_pretoken_file,
                pretoken_path = config.pretoken_path,
                output_paths=[config.vocab_path, config.merge_path]
                )
        print("finish bpe for tinystories.")

    if not os.path.exists(config.train_dataset_path):
        tokenizer = Tokenizer.from_files(vocab_filepath = config.vocab_path, 
                                        merges_filepath = config.merge_path, 
                                        special_tokens=config.special_tokens)
        tokenizer.encode_file(
            input_path = config.train_dataset_original_path, 
            output_path = config.train_dataset_path,
            num_split = 80,
            num_processes = 12
            )
        print("finish tokenization for train dataset of tinystories.")
        
    if not os.path.exists(config.train_dataset_path):
        tokenizer = Tokenizer.from_files(vocab_filepath = config.vocab_path, 
                                        merges_filepath = config.merge_path, 
                                        special_tokens=config.special_tokens)
        tokenizer.encode_file(
            input_path = config.valid_dataset_original_path, 
            output_path = config.train_dataset_path,
            num_split = 16,
            num_processes = 8
            )
        print("finish tokenization for train dataset of tinystories.")  
    
    print("finish init dataset")

if __name__ == "__main__":
    config = TrainingConfig(
        checkpoint_folder="./checkpoints/lr_1e-3/",
        learning_rate=1e-3,
        total_steps=3000,
        cosine_steps=3000,
        batch_size=64,
    )

    init_dataset(config)

    training(config)

    print("finish train for tinystories.")
    
