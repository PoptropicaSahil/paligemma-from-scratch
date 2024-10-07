import torch
from typing import List, Tuple


class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # NOTE: Shape of key_cache is (Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim)
            # So we return the sequence length
            return self.key_cache[0].shape[-2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV Cache of this layer, then create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Else concatenate the new keys with existing ones, along the sequence dimension
            # Each tensor has shape: (Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim)
            self.key_cache[layer_idx] = torch.cat(tensors=[self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(tensors=[self.value_cache[layer_idx], value_states], dim=-2)

        # Return all existing keys + new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
        # hidden_states: (Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim)
        
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape

        if num_repeats == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, num_repeats, seq_len, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * num_repeats, seq_len, head_dim)
        
        return hidden_states