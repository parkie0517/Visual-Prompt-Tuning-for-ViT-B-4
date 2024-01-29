# Visual_Prompt_Tuning_PyTorch
In this project, I applied Visual Prompt Tuning(VPT) method to the ViT-Base/4 model.  
I used a ViT-Base/16 model pre-trained on ImageNet-21k, 


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


## Implementation details
ViT-Base/4  
- Patch size: 4
- Embedding size: 768
- Hidden size: 768*4
- Head number: 12
- Layer depth: 12


## Results
The graphs below are training and testing accuracies.  
Orange line: ViT-Base/4 (Prompt Fine-tuning)
Blue Line: ViT-Base/4 (Full Fine-tuning)

![image](https://github.com/parkie0517/Visual-Prompt-Tuning-for-ViT-B-4/assets/80407632/1281b974-6f77-4e33-b500-2d9fbbcb118b)  
Training Accuracy

![image](https://github.com/parkie0517/Visual-Prompt-Tuning-for-ViT-B-4/assets/80407632/abe7befb-eea5-49e1-a53e-00c3079ab4d2)
Testing Accuracy

It took, 5m20s and 6m50s to train the prompted model and full fine-tuned model respectively.  
Although I have not trianed the models for a long time, I could observe that the prompted model convergene time was very short compared to the full fine-tuned model.  


## Conclusion
From the experiment results, I can conclude that VPT is a time and cost efficient method!
