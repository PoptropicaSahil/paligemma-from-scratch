from gemma import PaliGemmaForConditionalGeneration
from transformers import AutoTokenizer
from configs import PaliGemmaConfig
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import torch

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = 'right')
    assert tokenizer.padding_side == 'right'


    # find all *.safetensors files and load them in a dictionary
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    tensors = {} 
    for safetensor_file in safetensors_files:
        with safe_open(safetensor_file, framework="pt", device = "cpu") as f: #type: ignore
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    
    # load the model config from the official config json file
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        model_config = PaliGemmaConfig(**model_config_file)

    # Create the model with the loaded tensors
    model = PaliGemmaForConditionalGeneration(model_config).to(device)

    # load state dict of model
    # A state_dict is a Python dictionary object that maps each layer to its parameter tensor
    model.load_state_dict(tensors, strict = False)

    model.tie_weights()

    return model, tokenizer # type: ignore


def sample_top_p(probs: torch.Tensor, p: float = 0.95) -> torch.Tensor:
    # probs: (Batch_Size, Vocab_Size)

    # (Batch_Size, Vocab_Size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # (Batch_Size, Vocab_Size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Subtracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
    mask = (probs_sum - probs_sort) > p

    # Zero out all probabilities of tokens that are not selected by top p
    probs_sort[mask] = 0.0

    # Re-normalize the probabilities so that they sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample a token (its index) from top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # Get token position in the vocab corresponding to the sampled token
    # TODO: Check usage of torch.gather
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)

    return next_token