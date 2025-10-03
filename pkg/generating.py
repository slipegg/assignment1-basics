import torch

from model import softmax

def decoding(
        model: torch.nn.Module,
        input_id: torch.Tensor,
        max_seq_len: int,
        max_decode_length: int,
        temperature: float,
        top_k: int,
        end_token_id: int | None = None,
        device: str | None = None
):
    model.eval()
    generated_tokens = []
    
    input_ids = input_ids.to(device)
    model = model.to(device)

    with torch.no_grad():
        while len(generated_tokens) <= max_decode_length:
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[-max_seq_len:]
                print("Truncated input_ids to max_seq_len")
            logits = model(input_ids.unsqueeze(0))
            if temperature <= 0:
                next_token = logits[0, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=0)
            else:
                logits = logits[0, -1, :] / temperature
                softmax_logits = softmax(logits, dim=-1)
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(softmax_logits, top_k)
                    top_k_probs = top_k_values / top_k_values.sum()
                    next_token = top_k_indices[torch.multinomial(top_k_probs, num_samples=1)]
                else:
                    next_token = torch.multinomial(softmax_logits, num_samples=1)
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=0)
            
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    
    return generated_tokens
