# Visual_Prompt_Tuning_PyTorch
In this project, I applied Visual Prompt Tuning(VPT) method to the ViT-Base/4 model.  
I used a ViT-Base/16 model pre-trained on ImageNet-21k and used CIFAR-10 for fine-tuning.  
I compared two methods which were:
- Prompt fine-tune (97.83% in epoch 5!)
- Full fine-tune  


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
ViT-Base/16 was used. I had to modify the embedding layer so that CIFAR-10 could used for training.  
I modified version is called ViT-Base/4, where 4 denotes the patch size.  
Below are the implementation details of the ViT-Base/4 model.  
- Input image size: (3, 32, 32)
- Patch size: (3, 4, 4)
- Patch numbers: 64
- Prompt numbers: 50
- Embedding size: 768
- Hidden size: 768*4
- Head numbers: 12
- Layer depth: 12


## Experiment results
The graphs below are training and testing accuracies.  
Orange line: ViT-Base/4 (Prompt Fine-tuning)
Blue Line: ViT-Base/4 (Full Fine-tuning)

![image](https://github.com/parkie0517/Visual-Prompt-Tuning-for-ViT-B-4/assets/80407632/bee16f4e-f9a8-4109-bdfd-3877e31672fb)  
Training Accuracy

![image](https://github.com/parkie0517/Visual-Prompt-Tuning-for-ViT-B-4/assets/80407632/4588388f-6e0d-4208-b832-3af20f687b13)  
Testing Accuracy

It took, 5m20s and 6m50s to train the prompted model and full fine-tuned model respectively.  
Although I did not trian the models for sufficient amount of time, I could observe that the prompted model's convergene speed was very fast compared to the full fine-tuned model which did not even converge during 5 epochs.  


## Conclusion
From the experiment results, I can conclude that VPT is a time and cost efficient method!
