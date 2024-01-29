# Visual_Prompt_Tuning_PyTorch
Apply Visual Prompt Tuning (VPT) on a ViT model! Let's see if VPT improves the model's performance.

## Architecture of VPT for ViT-Base/4
![구조 이미지](https://github.com/parkie0517/Visual-Prompt-Tuning-for-ViT-B-4/assets/80407632/24c8f477-a7ed-4554-ad81-3601ce827546)  

※ Layers that are colored in Red(🔥) are intended to be trained, layers that are colored in Blue(❄️) are frozen!
- Input (3, 32, 32) image
- Divide into 64 patches
- Transform each patches into patch embeddings (768-dim) 
- Add positional embeddings to each patches and the classification token
- Add 50 prompts to the Transformer encoder's layers
- Apply Layer Normalization to the output of the encoder
- Input the classification token into the MLP head
- Output prediction!
