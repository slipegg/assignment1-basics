
import os
import pathlib

from cs336_basics.bpe.train_bpe import train_bpe
from cs336_basics.bpe.tokenizer import Tokenizer

if __name__ == "__main__":
    # 当前文件的父目录
    root_folder = pathlib.Path(__file__).resolve().parent.parent

    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-vocab.pkl'):
        train_bpe(f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'],
                False, f'{root_folder}/data/TinyStoriesV2-pretokens.pkl',
                [f'{root_folder}/data/TinyStoriesV2-vocab.pkl', f'{root_folder}/data/TinyStoriesV2-merges.pkl'])
        print("finish bpe for tinystories.")

    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy'):
        tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                        f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                        special_tokens=['<|endoftext|>'])
        tokenizer.encode_file(
            input_path = f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', 
            output_path = f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy',
            num_split = 80,
            num_processes = 12
            )
        print("finish tokenization for train dataset of tinystories.")

    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy'):
        tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                        f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                        special_tokens=['<|endoftext|>'])
        tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy')
        print("finish tokenization for valid dataset of tinystories.")
