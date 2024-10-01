import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from configs import PaliGemmaConfig


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__() # for nn.Module only :D
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
    

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache
    ):
        # (Batch_Size, Num_Patches, Hidden_Size)
        _, _, embed_dim = image_features.shape

        batch_size, sequence_length = input_ids.shape
        # dtype, device = input_embeds.dtype, input_embeds.device

        # (Batch_Size, Seq_Len, Hidden_Size)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
        # Size should be embedding dimension
        final_embedding = torch.zeros(size= (batch_size, sequence_length, embed_dim), dtype=input_embeds.dtype, device=input_embeds.device)
        # final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)

        # Text token is which is not image and padding :)
        # (Batch_Size, Seq_Len). True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

        # (Batch_Size, Seq_Len). True for image tokens
        image_mask = input_ids == self.config.image_token_index

        # (Batch_Size, Seq_Len). True for padding tokens
        pad_mask = input_ids == self.config.pad_token_id

        # Expand the masks to the embedding dimension
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, 1) -> (Batch_Size, Seq_Len, embed_dim)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)


        # Add the text embeddings
        final_embedding = torch.where(condition=text_mask_expanded, input=input_embeds, other=final_embedding)

        # Add image embeddings
        # NOTE: Usually there will be 256 tokens for image
        # NOTE: Cannot use torch.where because sequence length of scaled_image_features is not
        # equal to the sequence length of embedding
        final_embedding = final_embedding.masked_scatter(mask=image_mask_expanded, source=scaled_image_features)

        # Zero out padding tokens, because we aren't implementing it.
        final_embedding = torch.where(condition=pad_mask_expanded, input=torch.zeros_like(final_embedding), other=final_embedding)





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
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features=image_features, 
            input_embeds=input_embeds, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kv_cache = kv_cache
        )

        # Feed the sequence of inputs to language model
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache
        )

        return outputs