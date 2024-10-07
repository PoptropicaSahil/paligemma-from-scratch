import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from configs import GemmaConfig, PaliGemmaConfig
from kv_cache import KVCache


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings = 2048, base=10000 ,device=None):
                 
        super().__init__()
        # RoPE modify attention mechanism. Each head is independent and parallel. 
        # RoPE is applied to each head so works with head dim.
        self.dim = dim # set to the head_dim

        # max number of positions we can encode
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # calculate theta according to formula 
        # theta_i = base ^(2i/dim) where i=0,1,2,..., dim//2

        # self.inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.dim, step=2, dtype=torch.int64).float() / self.dim))
        # NOTE: Implementation here is different from paper due to HF choice
        # Original RoPE has pairs like t1, t1, t2, t2, ..., t_d/2, t_d/2
        # HF does like t1, t2, ... t_d/2, t1, t2, ... t_d/2
        # They had apparently already done some transform while concerting llama weights to hf
        # So now they had to perform an additional step. Overall effect is the same   
        self.inv_freq = self.base ** -(torch.arange(start=0, end=self.dim, step=2, dtype=torch.int64).float() / self.dim)
        self.register_buffer(name="inv_freq", tensor=self.inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        # Assuming size 1024, we are making like
        # [-513, -512, -511, ..., 0, 1, ..., 512]
        # Different than what is given in the paper but HF's way

        # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional embedding
        x1 = x[..., : x.shape[-1] // 2] # takes the first half of the last dimension
        x2 = x[..., x.shape[-1] // 2 :] # takes the second half of the last dimension
        return torch.cat((-x2, x1), dim = -1)
    
    @staticmethod
    def apply_rotary_pos_embedding(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim) # add head dim
        sin = sin.unsqueeze(unsqueeze_dim) # add head dim

        # Apply RoPE
        q_embed = (q*cos) + (GemmaRotaryEmbedding._rotate_half(q) * sin)
        k_embed = (k*cos) + (GemmaRotaryEmbedding._rotate_half(k) * sin)
        
        return q_embed, k_embed

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len = None):
        """Outputs will be the cos and sin vectors"""

        # x: (Batch_Size, num_attention_heads, seq_len, head_size)
        # TODO: position_ids: ()  
        self.inv_freq.to(x.device)

        # inv_freq_expanded: (Batch_Size, Head_Dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # position_ids_expanded: (Batch_Size, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        # mixed precision as a context manager, recommended for forward pass (not loss calculation though)
        with torch.autocast(device_type=device_type, enabled=False):
            
            # Multiply each theta by the position (which is the argument of the sin and cos function)
            # freqs: (Batch_Size, Head_Dim // 2, 1) @ (Batch_Size, 1, seq_len) -> (Batch_Size, Head_Dim // 2, seq_len) -> (Batch_Size, seq_len, Head_Dim // 2)
            # freqs are the arguments of the cosines and sines i.e. m*theta
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # self.inv_freq was till dim/2 but we need dim number of freqs
            # emb: (Batch_Size, seq_len, Head_Dim)
            emb = torch.cat((freqs, freqs), dim = -1)

            # (Batch_Size, seq_len, Head_Dim)
            cos, sin = emb.cos(), emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

