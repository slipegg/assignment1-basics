import torch

from cs336_basics.models.model import softmax
from cs336_basics.bpe.tokenizer import Tokenizer

def generate(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_seq_len: int,
        max_generate_length: int,
        temperature: float,
        top_k: int,
        end_token_id: int | None = None,
        device: str | None = None,
        tokenizer: Tokenizer =None,              # ✅ 新增参数
        stream_output: bool = True   # ✅ 是否实时打印
):
    model.eval()
    generated_tokens = []

    if tokenizer is not None and stream_output:
        prompt = tokenizer.decode(input_ids.tolist())
        print(prompt , end="", flush=True)
    
    input_ids = input_ids.to(device)
    model = model.to(device)
    with torch.no_grad():
        while len(generated_tokens) <= max_generate_length:
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[-max_seq_len:]
                # print("Truncated input_ids to max_seq_len")
            logits = model(input_ids.unsqueeze(0))  # shape: [1, seq_len, vocab_size]
            logits = logits[:, -1, :]               # 只取最后一个 token 的 logits，shape: [1, vocab_size]
            logits = logits.squeeze(0)              # 去掉 batch 维，shape: [vocab_size]

            if temperature <= 0:
                next_token = logits.argmax(dim=-1)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=0)
            else:
                logits = logits / temperature
                softmax_logits = softmax(logits, dim=-1)
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(softmax_logits, top_k)
                    top_k_probs = top_k_values / top_k_values.sum()
                    # next_token = top_k_indices[torch.multinomial(top_k_probs, num_samples=1)]
                    next_token = top_k_indices[torch.multinomial(top_k_probs, num_samples=1).item()]
                else:
                    # next_token = torch.multinomial(softmax_logits, num_samples=1)
                    next_token = torch.multinomial(softmax_logits, num_samples=1).squeeze(0)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=0)
            
            # ✅ 实时 decode + 输出
            if tokenizer is not None and stream_output:
                decoded_text = tokenizer.decode([next_token.item()])
                print(decoded_text, end="", flush=True)
            
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    
    return generated_tokens
