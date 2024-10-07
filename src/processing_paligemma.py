from typing import Dict, List
from PIL import Image
import torch
import numpy as np
from image_processing_utils import IMAGE_UTILS
from transformers import AutoTokenizer

# Huggingface uses 0.5! Actual values should be close but not equal
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaliGemmaProcessor:
    # Setup constant IMAGE_TOKEN
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer: AutoTokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens  # usually 256 image tokens
        self.image_size = image_size

        # NOTE: Tokenizer described in README.
        # We are extending the existing Gemma tokenizer with placeholder tokens
        # to handle incoming image from siglip
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # Extra tokens as defined by the article
        # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]

        # These tokens are used for object segmentation
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]

        tokenizer.add_tokens(EXTRA_TOKENS)

        # Get the token id of <image> token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # We will add BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        # Save the tokenizer
        self.tokenizer = tokenizer

    def __call__(
        self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True
    ) -> Dict:
        # Setting such that it works with one image and one prompt at a time
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images and {len(text)} prompts."

        utils = IMAGE_UTILS()
        # process images to tensors
        # handles rescale normalize etc
        pixel_values = utils.process_images(
            images=images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # List of numpy arrays -> single numpy array i.e. add a dimension corresponding to Batch_Size
        # (Batch_Size, Channel, Height, Width)
        pixel_values = np.stack(pixel_values, axis=0)

        # To torch tensor. One big tensor!
        pixel_values = torch.from_numpy(pixel_values)
        # pixel_values = torch.tensor(pixel_values)

        # Input to the model before tokenising
        # prepend a self.image_seq_length number of image tokens to the prompt
        # and tokenize the text ofcourse
        # README: Tokenizer Image - Blue marking and SentencePiece Tokenizer
        input_strings = [
            utils.add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenizer returns the input_ids and attn_mask for the input strings
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return {"pixel_values": pixel_values, **inputs}
