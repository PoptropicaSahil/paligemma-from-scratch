import torch
import torch.nn as nn
from typing import Tuple, Optional

class SiglipVisionConfig:
    """Configuration class"""

    def __init__(
        self,
        hidden_size=768,  # embeds size
        intermediate_size= 4* 768, # linear layer size. As expected
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,  # this paligemma supports 224*224 sized images
        patch_size=16,  # each patch 16*16
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,  # how many image embeds for each image!!
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size,

        self.intermediate_size = intermediate_size,
        self.num_hidden_layers = num_hidden_layers,
        self.num_attention_heads = num_attention_heads,
        self.num_channels = num_channels,
        self.image_size = image_size,
        self.patch_size = patch_size,
        self.layer_norm_eps = layer_norm_eps - 6,
        self.attention_dropout = attention_dropout,
        self.num_image_tokens = num_image_tokens



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        # hidden_size <==> embed_dim
        
        self.patch_embedding = nn.Conv2d(
            in_channels= config.num_channels, # 3 channels RGB
            out_channels= config.hidden_size,
            kernel_size= config.patch_size,
            stride= config.patch_size, # stride = patch size so no overlap!
            padding="valid" # shows no padding is added
        )

        # patch_size is 16, so embeds will be (img_size/ patch_size) * embed_dim
        self.num_patches = ( config.image_size //  config.patch_size) ** 2 # along length and breadth
        self.num_positions = self.num_patches # how many positional encodings -> one for each patch
        self.position_embedding = nn.Embedding(self.num_positions,  config.hidden_size) # NOTE: Learnable!
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor):
        # pixel_values: (Batch_Size, Channels, Height, Width)
        _, _, height, width = pixel_values.shape 

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, embed_dim, num_patches_H, num_patches_W)
        # num_patches_H = height // patch_size
        # num_patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)

        # (Batch_Size, embed_dim, num_patches_H, num_patches_W) -> (Batch_Size, embed_dim, num_patches_H * num_patches_W)
        embeddings = patch_embeds.flatten(2)

        # Get back in desired shape
        # (Batch_Size, embed_dim, num_patches_H * num_patches_W) -> (Batch_Size, num_patches_H * num_patches_W, embed_dim)
        embeddings = embeddings.transpose(1,2)

        # position_ids extracted from register buffer (just a range)
        # (Batch_Size, num_patches_H * num_patches_W, embed_dim) -> (Batch_Size, num_patches_H * num_patches_W, embed_dim)
        embeddings += self.position_embedding(self.position_ids)

        # (Batch_Size, num_patches_H * num_patches_W, embed_dim)
        # NOTE: num_patches_H * num_patches_W <==> Num_Patches
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple(torch.Tensor, Optional(torch.Tensor)):
        # takes in the output of first layer norm in encoder
        # NOTE: Num_Patches <==> seq_len as in language models(!!)       
        # hidden_states: (Batch_Size, Num_Patches, Embed_Dim)
        Batch_Size, Seq_Len, _ = hidden_states.size()

        # Each states shape (Batch_Size, Num_Patches, Embed_Dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Split for multi-head attention
        # Remember Num_Patches <=> Seq_Len
        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, num_heads, head_dim) -> (Batch_Size, num_heads, Num_Patches, head_dim)
        query_states = query_states.view(Batch_Size, Seq_Len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = query_states.view(Batch_Size, Seq_Len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = query_states.view(Batch_Size, Seq_Len, self.num_heads, self.head_dim).transpose(1,2)
        pass


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (Batch_Size, Num_Patches, Embed_Dim)

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Intermediate_Size)
        hidden_states = self.fc1(hidden_states)

        # (Batch_Size, Num_Patches, Intermediate_Size) -> (Batch_Size, Num_Patches, Intermediate_Size)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # (Batch_Size, Num_Patches, Intermediate_Size) -> (Batch_Size, Num_Patches, Embed_Dim)
        hidden_states = self.fc1(hidden_states)

        return hidden_states





class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: (Batch_Size, Num_Patches, Embed_Dim)
        residual = hidden_states

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states = self.layer_norm1(hidden_states)

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states += residual

        # (Batch_Size, Num_Patches, Embed_Dim)
        residual = hidden_states

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states = self.layer_norm2(hidden_states)

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states = self.mlp(hidden_states)

        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Embed_Dim) 
        hidden_states += residual

        return hidden_states





class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config) # get basic embeds for each patch
        self.encoder = SiglipEncoder(config) # will have attention
        self.post_layernorm = nn.LayerNorm(
            normalized_shape=self.embed_dim, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (Batch_Size, Channels, Height, Width) -> (Batch_Size, Num_Patches, Embed_Dim)
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> tuple:
        # After passing through vision model, we will have few patches (16), each of size embed_dim.
        # i.e. batch of list of embeddings
        # i.e. contextualized embeddings

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Num_Patches, Embed_Dim)
        return self.vision_model(pixel_values=pixel_values)
