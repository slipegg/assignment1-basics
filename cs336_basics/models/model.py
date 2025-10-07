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
        # weights @ x
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        

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
        
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * rms
        res = x_normed * self.weight
        return res.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, model_dim: int, feed_forward_dim: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(model_dim, feed_forward_dim, device=device, dtype=dtype)
        self.w2 = Linear(feed_forward_dim, model_dim, device=device, dtype=dtype)
        self.w3 = Linear(model_dim, feed_forward_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)).mul(self.w3(x)))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()

        half_dim = d_k // 2
        inv_freq = (theta ** -(torch.arange(0, half_dim, device=device).float() / half_dim)) # (half_dim,)

        index = torch.arange(int(max_seq_len*1.1), device=device)
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
        mask: Float[torch.Tensor, "... seq_len seq_len"] | None = None
) -> Float[torch.Tensor, "... d_v"]:
    d_k = Q.shape[-1]
    score = einsum(Q, K, "... seq_len d_k, ... seq_len_k d_k -> ... seq_len seq_len_k") / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    attention = softmax(score, dim=-1)
    return einsum(attention, V, "... seq_len seq_len_k, ... seq_len_k d_v -> ... seq_len d_v")


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                    is_use_rope: bool = False,
                    max_seq_len: int | None = None,
                    theta: float | None = None, 
                    device=None,
                    dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if is_use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        self.num_heads = num_heads
        
    def forward(self, x: torch.Tensor, 
                position: torch.Tensor | None = None
        ) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[-2]

        Q = rearrange(self.q_proj(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)

        if self.rope is not None:
            if position is None:
                position = torch.arange(seq_len, device=x.device)
                position = rearrange(position, 'seq_len -> 1 1 seq_len') # (batch=1, heads=1, seq_len)
            Q = self.rope(Q, position)
            K = self.rope(K, position)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)).expand(batch_size, self.num_heads, -1, -1)
        
        mha_attention_res = scaled_dot_product_attention(Q, K, V, mask=mask)
        mha_attention_res = rearrange(mha_attention_res, '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)')

        return self.output_proj(mha_attention_res)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, theta: float = 100000.0,  device=None, dtype=None):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, is_use_rope=True, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_res = x + self.attn(self.ln1(x))
        return att_res + self.ffn(self.ln2(att_res))


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size:int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float = 100000.0, device=None, dtype=None):
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: Int[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
