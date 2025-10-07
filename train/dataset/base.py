import torch
import torch.utils.data as tud
from typing import Literal

def role_template(message: str | list[dict[str, str]], role: Literal["system", "user", "assistant"]):
    if isinstance(message, str):
        message = [{"role": role, "content": message}]
    return message

class CollateDataset(tud.Dataset):
    def collate_fn(self,item_list):
        return torch.stack(item_list)
