import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from configs import PaliGemmaConfig


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__() # for nn.Module only
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Linear projection after vision encoder (Green in README image)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) 
        
        # Gemma model (VIOLET in README image)
        self.language_model = GemmaForCausalLM(config.text_config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id else -1

    
    def tie_weights(self):
        """Technique to reuse params of one layer to another in LLMs"""
        return self.language_model.tie_weights()
    

    def forward(
            self,
            input_ids: torch.LongTensor = None, # input ids by tokenizer in paligemma processor
            pixel_values: torch.LongTensor = None, # image loaded by paligemma processor (Batch_Size, Channel, Height, Width)
            attention_mask: Optional[torch.Tensor] = None, # attention mask by tokenizer in paligemma processor
            kv_cache: Optional[KVCache] = None
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded" # we haven't implemented

        # Extract the input embeddings
        # (Batch_Size, Seq_Len, Hidden_Size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Merge text and images. Vision model extracts (contextualized) embeddings from the image
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Num_Patches, Embed_Dim)
        selected_image_feature = self.vision_tower(pixel_values = pixel_values.to(input_embeds.dtype))

        # Convert embeddings to size such that language model can take it
        # Will be concatenated with the text tokens
        # (Batch_Size, Num_Patches, Embed_Dim) -> (Batch_Size, Num_Patches, Hidden_Size)
        # NOTE: Hidden_Size is same as the embedding size for the language model 
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)  

        # Feed the sequence of inputs to language model
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache
        )

        return outputs