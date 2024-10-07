import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from configs import GemmaConfig, PaliGemmaConfig
from kv_cache import KVCache
from rope import GemmaRotaryEmbedding


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
    """
    NOTE: Made up of many layers, each layer will have its own KV cache.
    Therefore we will have to pass layer's index to check which KV cache should be used
    Gemma model has 1 head for K, Q and 8 heads for V -> Multi-query attention
    """
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

        self.q_proj = nn.Linear(in_features = self.hidden_size, out_features = self.num_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(in_features = self.hidden_size, out_features = self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(in_features = self.hidden_size, out_features = self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(in_features = self.num_heads * self.head_dim, out_features = self.hidden_size, bias = config.attention_bias)

        self.rotary_embeds = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # hidden_states: (Batch_Size, Seq_Len, Hidden_Size)
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  # (Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim)
        key_states = self.k_proj(hidden_states)  # (Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim)
        value_states = self.v_proj(hidden_states)  # (Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim)

        # (Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim) -> (Batch_Size, Seq_Len, Num_Heads_Q, Head_Dim) -> (Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)  
        
        # (Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim) -> (Batch_Size, Seq_Len, Num_Heads_KV, Head_Dim) -> (Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim)        
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  

        # Apply RoPE
        # NOTE: See that after apply_rotary_pos_embedding, we get back the query states and key states in the same shape
        # Hence, we are only adding information to them

        # (Batch_Size, Seq_Len, Head_Dim), (Batch_Size, Seq_Len, Head_Dim)
        cos, sin = self.rotary_embeds(value_states, position_ids, seq_len=None)

        # (Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim), (Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim)
        query_states, key_states = GemmaRotaryEmbedding.apply_rotary_pos_embedding(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states=key_states, value_states=value_states, layer_idx=self.layer_idx)

        # Repeat keys and values to match number of heads of the query
        key_states = KVCache.repeat_kv(hidden_states=key_states, num_repeats=self.num_key_value_groups)
        value_states = KVCache.repeat_kv(hidden_states=value_states, num_repeats=self.num_key_value_groups)

        # TODO: Check the shapes. Why is Seq_Len different for KV and Q?
        # (Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim) @ (Batch_Size, Num_Heads_KV, Head_Dim, Seq_Len) -> (Batch_Size, Num_Heads_Q, Seq_Len, Seq_Len)
        attn_weights = torch.matmul(input=query_states, other=key_states.transpose(2,3) / math.sqrt(self.head_dim))
        
        assert attention_mask is not None

        attn_weights = attn_weights + attention_mask

        # Apply softmax
        # (Batch_Size, Num_Heads_Q, Seq_Len, Seq_Len_KV)
        attn_weights = nn.functional.softmax(input=attn_weights, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
        attn_weights = nn.functional.dropout(input=attn_weights, p=self.attention_dropout, training=self.training)

        # TODO: Check the shapes
        # (Batch_Size, Num_Heads_Q, Seq_Len, Seq_Len_KV) @ (Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim) -> (Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim)
        attn_output = torch.matmul(input=attn_weights, other=value_states)


        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of the size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}" 
            )
    
        # Get in original shapes
        # (Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim) -> (Batch_Size, Seq_Len, Num_Heads_Q, Head_Dim) -> (Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim)
        # Num_Heads_Q * Head_Dim equivalent to hidden_dim
        attn_output  = attn_output.transpose(1, 2).view(batch_size, q_len, -1)

        # (Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim) -> (Batch_Size, Seq_Len, Hidden_Size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights # type: ignore


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