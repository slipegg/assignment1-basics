import time
import regex as re
import pickle
import os
import pathlib
import functools
import numpy as np
from abc import ABC
import multiprocessing
import heapq
from collections.abc import Iterable, Iterator
from cs336_basics.bpe.tokenizer import Tokenizer

if __name__ == "__main__":
    root_folder = pathlib.Path(__file__).resolve().parent.parent

    tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                     f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                     special_tokens=['<|endoftext|>'])

    encode_target_file = f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt'
    encode_res_file = f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy'
    tokenizer.encode_file(encode_target_file, encode_res_file)
    # 读取.npy文件
    token_ids = np.load(encode_res_file).tolist()
    print(token_ids[:100])
    print(tokenizer.decode(token_ids[:100]))
    # tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy')
    # tokenizer.encode_file(f'{root_folder}/data/corpus.en', f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy')
    
