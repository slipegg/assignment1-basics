import torch
import torch.nn as nn
import math

from einops import einsum, rearrange

def initialize_linear(layer):
    """
    初始化 Linear 层
    N(μ=0, σ²=2/(d_in+d_out)) truncated at [-3σ, 3σ]
    """
    d_in = layer.in_features
    d_out = layer.out_features
    
    # 计算标准差
    std = math.sqrt(2.0 / (d_in + d_out))
    
    # 使用截断正态分布初始化权重
    nn.init.trunc_normal_(
        layer.weight,
        mean=0.0,
        std=std,
        a=-3*std,  # 下界
        b=3*std    # 上界
    )
    
    # 如果有偏置项，也进行初始化（通常设为0）
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def initialize_embedding(layer):
    """
    初始化 Embedding 层
    N(μ=0, σ²=1) truncated at [-3, 3]
    """
    nn.init.trunc_normal_(
        layer.weight,
        mean=0.0,
        std=1.0,
        a=-3.0,
        b=3.0
    )

def initialize_rmsnorm(layer):
    """
    初始化 RMSNorm 层
    所有参数设为 1
    """
    nn.init.ones_(layer.weight)


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
        
        