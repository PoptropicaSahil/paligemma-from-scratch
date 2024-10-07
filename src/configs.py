from typing import Optional, Dict

"""
We will load most of the config params from the config json files from HF
We could have used dataclasses to store these config classes, however, handling of
kwargs is not yet nicely supported with datalclasses, so we stick to the usual notation.
"""


class SiglipVisionConfig:
    """Configuration class for Siglip"""

    def __init__(
        self,
        hidden_size: int = 768,  # embeds size
        intermediate_size: int = 4 * 768,  # linear layer size. As expected
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,  # this paligemma supports 224*224 sized images
        patch_size: int = 16,  # each patch 16*16
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,  # how many image embeds for each image!!
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class GemmaConfig:
    """Like the config of any other LLM"""

    def __init__(
        self,
        vocab_size,
        hidden_size,  # embed size
        intermediate_size,  # intermediate size in the FF layer
        num_hidden_layers,  # num layers
        num_attention_heads,  # num heads for queries
        num_key_value_heads,  # for grouped query attention
        head_dim=256,  # dim of each head
        max_position_embeddings=8192,  # max num of position the model has been trained upon - useful for RoPE
        rms_norm_eps=1e-6,
        rope_theta=10000.0,  # theta for RoPE, the base frequency
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    """Paligemma is the entire model: SigLIP + Tokenizer + Layer + Gemma
    So this Config needs info about everything
    That is why we have vision_config and text_config being passed
    """

    def __init__(
        self,
        vision_config: Dict,
        text_config: Dict,
        ignore_index=-100,
        image_token_index=256000,  # token corresponding to placeholder token <image>
        vocab_size=257152,
        projection_dim=2048,  # Output size of Linear Projection layer of SigLIP. i.e. dimension that image features should be resized to before feeding to Gemma
        hidden_size=2048,  # embedding size of language model
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False  # needed by HF
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # NOTE: CHECK - Number of image tokens in the linear layer will be number of patches **2 ?
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
