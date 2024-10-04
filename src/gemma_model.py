import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from configs import GemmaConfig, PaliGemmaConfig


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        """NOTE: Anything on HF like ...ForCausalLM is like a model with a linear head (convert each embed to logits)"""
        super().__init__()
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds: (Batch_Size, Seq_Len, Hidden_Size)

        # outputs: (Batch_Size, Seq_Len, Hidden_Size)
        outputs = self.model(
            attention_mask=attention_mask, position_ids=position_ids, input_embeds=input_embeds, kv_cache=kv_cache
        )
        # outputs from model equivalent to hidden_states
        logits = self.lm_head(outputs)
        logits = logits.float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            # return updated cache
            return_data["kv_cache"] = kv_cache

        return return_data


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x)
        # NOTE: Llama does x.to(float16) * w
        # Gemma does (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        """Just a bit modified version of usual MLP"""
        # x: (Batch_Size, Seq_Len, Hidden_Size)

        # (Batch_Size, Seq_Len, Hidden_Size) -> (Batch_Size, Seq_Len, Intermediate_Size)
        y = self.gate_proj(x)
        y = nn.functional.gelu(y, approximate='tanh')

        # (Batch_Size, Seq_Len, Hidden_Size) -> (Batch_Size, Seq_Len, Intermediate_Size)
        j = self.up_proj(x)

        # (Batch_Size, Seq_Len, Intermediate_Size) 
        z = y * j 
        # (Batch_Size, Seq_Len, Intermediate_Size) -> (Batch_Size, Seq_Len, Hidden_Size)
        z = self.down_proj(z)



class GemmaAttention(nn.Module):
    """NOTE: Made up of many layers, each layer will have its own KV cache"""
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # pass layer index so that we can pass the KV cache correctly
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # number of heads for queries
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads # number of heads for keys and values
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings # how many positions can we encode via rope
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.hidden_size = config.hidden_size
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # hidden_states: (Batch_Size, Seq_Len, Hidden_Size)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config = config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layer_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Similar to SiglipLayers"""

        residual = hidden_states

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = self.input_layer_norm(hidden_states)

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = residual + hidden_states

        # (Batch_Size, Seq_Len, Hidden_Size)
        residual = hidden_states

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = self.post_attn_layernorm(hidden_states)

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = self.mlp(hidden_states)

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = residual + hidden_states

        return hidden_states




class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = input_embeds

        # (Batch_Size, Seq_Len, Hidden_Size)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for layer in self.layers:
            # (Batch_Size, Seq_Len, Hidden_Size)
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache
            )

        # (Batch_Size, Seq_Len, Hidden_Size)
        hidden_states = self.norm(hidden_states)

        # (Batch_Size, Seq_Len, Hidden_Size)
        return hidden_states