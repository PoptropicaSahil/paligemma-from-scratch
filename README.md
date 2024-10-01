# Coding Vision Language Model from Scratch


Like my previous implementation, this repo is a code follow-along excercise by the [excellent tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw) by Umar Jamil `@hkproj`. 

``` bash
The code covers
- Vision Transformer
- Contrastive Learning (CLIP and SigLIP)
- Language Model (Gemma)
- KV Cache
- Rotary Positional Encoding (RoPE)
- Normalization (Batch, Layer, RMS)
```

![alt text](readme-images/intro.png)

## Why PaliGemma
* PaliGemma achieves state-of-the-art results across various vision-language tasks, including image captioning, visual question answering (VQA), and specialized tasks like chart understanding and optical character recognition (OCR) [reference](https://syncedreview.com/2024/07/26/from-images-to-insights-deepminds-versatile-vision-language-model-paligemma-achieves-sota-results/). 
* Plaigemma also has few-show learning capabilities.
* Only 3B params - fairly lightweight!


## Advantages over CLIP

| Feature                | PaliGemma                                    | CLIP                                      |
|------------------------|:--------------------------------------------:|:------------------------------------------:|
|**Image Encoder**       | SigLIP                                       | Jointly trained image encoder             |
|**Text Decoder**        | Gemma                                        | No dedicated text decoder                 |
| **Loss Function!!**      | **Sigmoid <br>(computationally cheaper)**   | **Softmax cross-entropy  <br>(more resource-intensive)** |
| **Output Types**       | Text outputs from images	                    | Generates embeddings                      |
| **Task Performance**   | Strong in object detection and segmentation  | Excels in zero-shot classification, not detection?!         |
| **Open Source**        | Yes                                          | No                                        |                
| **Training Datasets**  | vision-language datasets                     | image-text pairs                          |

> Directly training in contrastive fashion is better because we can download billions of images from the internet. Mostly, images are captioned, like the alt_text atleast :)

## CLIP Basics
Loss is the average loss over rows and columns (`axis=0` and `axis=1`). Because the matrix is not symmetric. Like $I_1 \cdot T_0 \neq T_1 \cdot I_0 $
![alt text](readme-images/clip.png)

But the softmax function (loss when calculating over every row and column) is *numerically unstable*. We have to multiply by this constant to make it stable. \
This will have to be done for every row every column. Plus, the usual exponentials, sum of all exponentials etc. **Too much computation** \
Even while parallelising, we have to keep a full row or full col in a memory. So have to keep batch size less
![alt text](readme-images/softmax-unstable1.png)

![alt text](readme-images/softmax-unstable2.png)

**Solution - Sigmoid loss!**

Don't worry about each row and each col. Instead, treat it as a binary classification task. Over each item in the all-dot-products matrix. 

Independently over all other items. Much more parallelisable! No need for normalization constants.

Labels for each item will be either 1 (if on diagonal) or 0 (if off diagonal).
![alt text](readme-images/siglip.png)



## Vision Transformer
It is a **seq-to-seq** model. 

> **Input** -> collection of embeddings. For 16 patches, we have 16 embeddings (obtained by convolution over that patch). Each patch's embedding only has info about its pixels. (*ofc*)

> **Output** -> Contextualized embeddings i.e. series of embeddings that has patch info + position info + info about context 

![alt text](readme-images/vision1.png)

Difference from the usual language models
-  Positional embeddings are not sinusoids here. They are learnable ones. 
- In language models, contextual embeddings are created by causal masking i.e. attention uses causal mask. The output embeddings in vision are not masked in any way. Image patches should have knowledge about nearby patches no probs :D Images aren't autoregressive.



## Normalization 101
Main reason - **covariate shift**. 

One batch of inputs, say values range (1-10) will give next layer outputs in similar range (assuming weights are 0-1). Say next batch of inputs ranges in (100-200), then outputs also shoot up.

$\implies$ Big change in loss while training
$\implies$ Big change in gradient
$\implies$ Big change in weights training
$\implies$ Slow training

Solution 1: **Batch Normalization** <br>
We calulate statistics along each dimension of the vector representing an item. For this to work well, usually need a large batch size, so that the $\mu, \sigma$ for each dim stabalise enough a lot of samples. But still faster than earlier. 
![alt text](readme-images/batch-norm.png)


Solution2: **Layer Normalization** <br>
Calculate the statistics along each row, makes training more stable, because don't need large batch size now. 
![alt text](readme-images/layer-norm.png)


## Encoders
Remember, encoders (like BERT) are seq-to-seq models. For each input, you get an embedding. 
> Implemented in the `class SiglipEncoderLayer` in `modelling_siglip.py`<br> 
![alt text](readme-images/encoder.png)

```
Note the two major objectives of multi-head attention
1. All the heads run in parallel. 
2. Each head learns to relate tokens/patches differently
```

Now that we have a collection of contextualized embeddings for each image, we can take the mean to get a single embedding. For language tasks, models like `CLIP` took the embedding of the `CLS` token as a representative of the whole sentence. 


## Tokenizer ([official repo](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma#tokenizer)). Also read [Detailed Inference Process](https://huggingface.co/blog/paligemma#detailed-inference-process)
PaliGemma uses the Gemma tokenizer with 256'000 tokens, but we **further extend its vocabulary** with 1024 entries that represent coordinates in normalized image-space (<loc0000>...<loc1023>), and another with 128 entries (<seg000>...<seg127>) that are codewords used by a lightweight referring-expression segmentation vector-quantized variational auto-encoder (VQ-VAE) with the architecture of Ning et al. (2023) and trained on OpenImages as in PaLI-3. While the big_vision codebase is flexible enough to extend tokenizers on-the-fly, we also provide a SentencePiece model file of the Gemma tokenizer with these additional tokens baked in, for the convenience of other codebases.


> The input text is tokenized normally, and a `<bos>` token is added at the beginning, and an additional newline token `\n` is appended. The tokenized text is also prefixed with a fixed number of `<image>` tokens. The model uses full block attention for the **complete input `(image + <bos> + prompt + \n)`**, and a causal attention mask for the generated text.



Gemma tokenizer will ofcourse generate tokens for the text. But we also need tokens for the image. So we insert some placeholder tokens `<image>` that will then be replaced with the embeddings extracted by the Vision Encoder. 
![alt text](readme-images/extending-tokenizer.png)


We are using the [paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) model. Pre-trained with `224*224 input images` and `128 token input/output text` sequences.

![alt text](readme-images/tokenization.png)
Paper says '\n' token be tokenized seperately. But HF implementation does not! 


## Weight Tieing
Technique to reuse params of one layer to another in LLMs (Decoder only part :D). Since the last linear layer and the initial layer perform reverse functions, their weights cab be reused for efficiency i.e. will have to work with less number of parameters.
![alt text](readme-images/tie-weights.png)


## KV Cache
