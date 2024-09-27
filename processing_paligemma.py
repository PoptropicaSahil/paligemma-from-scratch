import token
from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaliGemmaProcessor:

    # Setup constant IMAGE_TOKEN
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_lenght = num_image_tokens
        self.image_size = image_size

        # NOTE: Tokenizer described in README.
        # We are extending the existing Gemma tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        
        # These tokens are used for object segmentation
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # We will add BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        # Save the tokenizer
        self.tokenizer = tokenizer