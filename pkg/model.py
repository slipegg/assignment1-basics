import torch
import torch.nn as nn
import math

from einops import einsum, rearrange
from jaxtyping import Float, Int


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        weight = torch.empty(out_features, in_features, 
                                    device=device, dtype=dtype)
        
        # 按要求初始化权重
        ## 计算标准差
        std = math.sqrt(2.0 / (in_features + out_features))
        
        ## 使用截断正态分布初始化权重
        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=std,
            a=-3*std,  # 下界
            b=3*std    # 上界
        )

        self.weight = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ weight.T
        return einsum(x, self.weight, "... in_features, in_features out_features -> ... out_features")
        

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        # 按要求初始化权重
        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
        

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.weight = torch.ones(d_model, device=device, dtype=dtype)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x_normed = x / rms
        res = x_normed * self.weight
        return res.to(in_dtype)
    

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, model_dim: int, feed_forward_dim: int, device=None, dtype=None):
        super().__init__()

        self.linear_1 = Linear(model_dim, feed_forward_dim, device=device, dtype=dtype)
        self.linear_2 = Linear(feed_forward_dim, model_dim, device=device, dtype=dtype)
        self.linear_3 = Linear(model_dim, feed_forward_dim, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.silu(self.linear_1(x)).mul(self.linear_3(x)))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        half_dim = d_k // 2
        inv_freq = (theta ** -(torch.arange(0, half_dim, device=device).float() / half_dim)) # (half_dim,)

        index = torch.arange(max_seq_len, device=device)
        theta_table = torch.outer(index, inv_freq)  # (max_seq_len, half_dim)
        self.register_buffer('cos', torch.cos(theta_table), persistent=False)
        self.register_buffer('sin', torch.sin(theta_table), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) 位置索引
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # x1: [..., seq_len, 0::2]
        # x2: [..., seq_len, 1::2]
        x1, x2 = rearrange(x, '... (half_dim x1x2) -> x1x2 ... half_dim', x1x2=2)

        res1 = x1 * cos - x2 * sin
        res2 = x1 * sin + x2 * cos
        return torch.stack((res1, res2), dim=-1).flatten(start_dim=-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.amax(dim=dim, keepdim=True)
    x_exp = (x - x_max).exp()
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, "... seq_len d_k"],
        K: Float[torch.Tensor, "... seq_len d_k"],
        V: Float[torch.Tensor, "... seq_len d_v"],
        mask: Float[torch.Tensor, "seq_len seq_len"] | None = None
) -> Float[torch.Tensor, "... d_v"]:
    d_k = Q.shape[-1]
    score = einsum(Q, K, "... seq_len d_k, ... seq_len d_k -> ... seq_len seq_len") / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    attention = softmax(score, dim=-1)
    return einsum(attention, V, "... seq_len seq_len, ... seq_len d_v -> ... seq_len d_v")


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                    is_use_rope: bool = False,
                    theta: float | None = None, 
                    max_seq_len: int | None = None,
                    device=None,
                    dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_weight = Linear(d_model, d_model, device=device, dtype=dtype)

        if is_use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        self.num_heads = num_heads
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[-1], x.shape[-2]

        Q = rearrange(self.q.weight(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K = rearrange(self.k.weight(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        V = rearrange(self.v.weight(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)

        if self.rope is not None:
            position = torch.arange(seq_len, device=x.device)
            position = rearrange(position, 'seq_len -> 1 1 seq_len') # (batch=1, heads=1, seq_len)
            Q = self.rope(Q, position)
            K = self.rope(K, position)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        mha_attention_res = scaled_dot_product_attention(Q, K, V, mask=mask)
        mha_attention_res = rearrange(mha_attention_res, '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)')

        return self.out_weight(mha_attention_res)


class transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 100000.0, max_seq_len: int = 2048, device=None, dtype=None):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_use_rope=True, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_res = x + self.attention(self.norm1(x))
        return att_res + self.ffn(self.norm2(att_res))


class transformer_llm(torch.nn.Module):
    def __init__(self, vocab_size:int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float = 100000.0, device=None, dtype=None):
        super().__init__()
        
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: Int[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        x = self.token_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_norm(x)
        logits = self.output_linear(x)
        return logits
        