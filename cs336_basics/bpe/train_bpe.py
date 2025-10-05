import time
import regex as re
from collections import Counter, defaultdict
import multiprocessing
import functools
import pickle

from cs336_basics.utils.priority_dict import PriorityDict
from cs336_basics.utils.pretokenization import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initialize_occurrence_pair(vocab:dict[tuple[bytes], int]) -> tuple[defaultdict[tuple[bytes], int],
                                                                       defaultdict[tuple[bytes], set[tuple[bytes]]]]:
    """
    Initialize the occurrence pair dictionary.
    """
    pairsfreq = PriorityDict()
    pair2words = defaultdict(set)
    for words, freq in vocab.items():
        for i in range(len(words)-1):
            pairsfreq[words[i], words[i+1]] += freq
            pair2words[words[i], words[i+1]].add(words)
    return pairsfreq, pair2words

def process_chunk(chuck: tuple[int],
                  input_path: str,
                  special_tokens: list[str]) -> Counter:
    """
    Process each chunk of the file and update the vocabulary counter.
    """
    start, end = chuck
    special_tokens_pattern = '|'.join(special_tokens)
    # special_tokens_pattern = "|".join(map(re.escape, special_tokens))
    chunk_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 2.1 在预分词前移除特殊标记
        for segment in re.split(special_tokens_pattern, chunk):
            # 3. 预分词(pre-tokenization)
            for match in re.finditer(PAT, segment):
                if match.group():
                    chunk_counter.update([tuple(bytes([b]) for b in match.group().encode('utf-8'))])
    return chunk_counter

# 维护三个数据结构
# 1. vocab_counter: word的词频数目
# 2. occur_pair_freq: pair的词频数目
# 3. pair2words: pair对应的word
def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str],
              use_pretoken_file: bool = False,
              pretoken_file: str | None = None,
              bpe_results_file: list[str] | None = None
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 0. 初始变量
    num_processes = 2
    # 1. 构建初始词汇表
    vocab : dict[int, bytes] = { k: bytes([k]) for k in range(256) }
    for (i, special_token) in enumerate(special_tokens):
        vocab.update( { 256+i: special_token.encode() } )
    initial_vocab_size = len(vocab)
    merges : list[tuple[bytes, bytes]] = []
    vocab_counter : dict[tuple[bytes], int] = Counter()

    # 2. 预分词(pre-tokenization)
    t0 = time.time()
    t1 = time.time()
    if use_pretoken_file:
        t1 = time.time()
        with open(pretoken_file, "rb") as f:
            vocab_counter = pickle.load(f)
    else:
        boundaries = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, "<|endoftext|>".encode("utf-8"))
        t1 = time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(
                    functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
                    zip(boundaries[:-1], boundaries[1:]),
                )

            for res in results:
                vocab_counter.update(res)
                        
        if pretoken_file:
            with open(pretoken_file, "wb") as f:
                pickle.dump(vocab_counter, f)

    t2 = time.time()
    # 3. 训练BPE
    def update_vocab(pair):
        new_elem = (pair[0])+(pair[1])
        vocab.update({len(vocab): new_elem})
        merges.append(pair)

    occur_pair_freq, pair2words = initialize_occurrence_pair(vocab_counter)
    
    # def merge_pair(word: tuple[bytes], pair:tuple[bytes], pair_bytes: bytes, freq: int) -> tuple[bytes]:
    #     ret = []
    #     i = 0
    #     n = len(word)
    #     new_pairs = []
    #     prev_elem = None
    #     curr_elem = None
    #     while i < n:
    #         # 删除旧word
    #         if i < n - 1:
    #             pair2words[(word[i], word[i+1])].discard(word)

    #         if i < n - 1 and (word[i], word[i+1]) == pair:
    #             if i > 0:
    #                 occur_pair_freq[word[i-1], word[i]] -= freq
    #                 occur_pair_freq[word[i-1], pair_bytes] += freq
    #             if i < n - 2:
    #                 occur_pair_freq[word[i+1], word[i+2]] -= freq
    #                 occur_pair_freq[pair_bytes, word[i+2]] += freq
    #             curr_elem = pair_bytes
    #             i += 2
    #         else:
    #             curr_elem = word[i]
    #             i += 1

    #         ret.append(curr_elem)
    #         if prev_elem:
    #             new_pairs.append((prev_elem, curr_elem))
    #         prev_elem = curr_elem

    #     new_word = tuple(ret)
    #     for p in new_pairs:
    #         pair2words[p].add(new_word)
    #     return new_word

    def merge_pair(word: tuple[bytes], pair:tuple[bytes], pair_bytes: bytes, freq: int) -> tuple[bytes]:
        ret = []
        i = 0
        n = len(word)
        new_pairs = []
        prev_elem = None
        curr_elem = None
        
        while i < n:
            # 删除旧word对应的pair
            if i < n - 1:
                old_pair = (word[i], word[i+1])
                pair2words[old_pair].discard(word)
                # 如果这个pair没有对应的words了，并且频率为0，就移除它
                if not pair2words[old_pair]:
                    # 我们不能直接从occur_pair_freq中删除，因为PriorityDict不支持
                    # 但我们可以将其频率设为0，这样在pop时会被忽略
                    if old_pair in occur_pair_freq:
                        occur_pair_freq[old_pair] = 0  # 设为0，让pop逻辑处理

            if i < n - 1 and (word[i], word[i+1]) == pair:
                # 更新相邻pair的频率
                if i > 0:
                    left_pair = (word[i-1], word[i])
                    if left_pair in occur_pair_freq:
                        occur_pair_freq[left_pair] -= freq
                        # 如果频率变为0或负数，设为0让pop逻辑处理
                        if occur_pair_freq[left_pair] <= 0:
                            occur_pair_freq[left_pair] = 0
                    
                    new_left_pair = (word[i-1], pair_bytes)
                    # 修复：使用安全的get方式
                    if new_left_pair in occur_pair_freq:
                        occur_pair_freq[new_left_pair] += freq
                    else:
                        occur_pair_freq[new_left_pair] = freq
                        
                    if new_left_pair not in pair2words:
                        pair2words[new_left_pair] = set()
                    pair2words[new_left_pair].add(word)
                    
                if i < n - 2:
                    right_pair = (word[i+1], word[i+2])
                    if right_pair in occur_pair_freq:
                        occur_pair_freq[right_pair] -= freq
                        if occur_pair_freq[right_pair] <= 0:
                            occur_pair_freq[right_pair] = 0
                        
                    new_right_pair = (pair_bytes, word[i+2])
                    # 修复：使用安全的get方式
                    if new_right_pair in occur_pair_freq:
                        occur_pair_freq[new_right_pair] += freq
                    else:
                        occur_pair_freq[new_right_pair] = freq
                        
                    if new_right_pair not in pair2words:
                        pair2words[new_right_pair] = set()
                    pair2words[new_right_pair].add(word)
                    
                curr_elem = pair_bytes
                i += 2
            else:
                curr_elem = word[i]
                i += 1

            ret.append(curr_elem)
            if prev_elem:
                new_pairs.append((prev_elem, curr_elem))
            prev_elem = curr_elem

        new_word = tuple(ret)
        
        # 更新新word对应的pairs
        for p in new_pairs:
            if p not in pair2words:
                pair2words[p] = set()
            pair2words[p].add(new_word)
            # 确保pair在occur_pair_freq中存在
            if p in occur_pair_freq:
                occur_pair_freq[p] += freq
            else:
                occur_pair_freq[p] = freq
            
        return new_word

    print("occur_pair:", len(occur_pair_freq))
    print("vocab_counter:", len(vocab_counter))
    t3 = time.time()
    best_pair, _ = occur_pair_freq.pop()
    while len(vocab) < vocab_size:
        # print("vocab size:", len(vocab))
        
        if not occur_pair_freq:
            print("No more pairs to merge")
            break
            
        best_pair, best_freq = occur_pair_freq.pop()
        
        # 检查best_pair是否仍然有效
        if best_pair not in pair2words or not pair2words[best_pair]:
            continue
            
        new_vocab = {}
        del_keys = set()

        best_pair_bytes = best_pair[0] + best_pair[1]
        words_to_merge = pair2words.pop(best_pair)
        
        for word in words_to_merge:
            if word not in vocab_counter:  # 确保word仍然存在
                continue
                
            freq = vocab_counter[word]
            new_word = merge_pair(word, best_pair, best_pair_bytes, freq)
            new_vocab[new_word] = freq
            del_keys.add(word)

        # 更新vocab_counter
        for k in del_keys:
            if k in vocab_counter:
                del vocab_counter[k]
        vocab_counter.update(new_vocab)
        
        update_vocab(best_pair)

    t4 = time.time()
    print("read = ", t1-t0, ", pre-tokenization = ", t2-t1, ", pre-train = ", t3-t2, ", merge = ", t4-t3)

    
    if bpe_results_file:
        with open(bpe_results_file[0], "wb") as f:
            pickle.dump(vocab, f)
        with open(bpe_results_file[1], "wb") as f:
            pickle.dump(merges, f)

    return vocab, merges