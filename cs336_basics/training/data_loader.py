import torch
import numpy.typing as npt
import numpy as np

# def data_loading(
#         x: npt.NDArray, 
#         batch_size: int, 
#         context_length: int, 
#         device: str
#     )-> tuple[torch.Tensor, torch.Tensor]:
#     start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
#     all_indices = start_indices[:, None] + np.arange(context_length + 1)[None, :]

#     inputs = x[all_indices[:, :-1]]
#     targets = x[all_indices[:, 1:]]

#     return tuple([torch.from_numpy(t).to(device) for t in (inputs, targets)])
def data_loading(
        x: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    all_indices = start_indices[:, None] + np.arange(context_length + 1)[None, :]

    inputs = x[all_indices[:, :-1]]
    targets = x[all_indices[:, 1:]]

    # 确保输出为 int64 类型的张量
    return tuple([torch.from_numpy(t).to(device).long() for t in (inputs, targets)])