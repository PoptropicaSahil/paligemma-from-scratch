from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from kv_cache import KVCache
from gemma import PaliGemmaForConditionalGeneration
from utils import load_hf_model


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = False,
    only_cpu: bool = False
): 
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda" 
        elif torch.backends.mps.is_available():
            device = "mps"
        
    print(f'Device in use = {device}')

    print(f'Loading model ...')

    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size

    processor = PaliGemmaProcessor(tokenizer=tokenizer, num_image_tokens=num_image_tokens, image_size=image_size)


    print(f'Running inference ...')

    with torch.no_grad():
        test_inference(model, processor, device, prompt, image_file_path, 
                       max_tokens_to_generate, temperature, top_p, do_sample)