import torch
import torch.nn as nn
import math
import timm

class CustomPrompts(nn.Module):
    """
    CustomPrompts class is for the following purposes
        1. Create, initialize, and store propt embeddings
        2. Incorporate prompts into input embeddings
    """
    def __init__(self, num_prompts=50, prompt_dim=768, num_layers=12):
        """
        Create, initialize, and store prompt embeddings
        """
        super().__init__()
        self.num_prompts = num_prompts
        # Calculate the value for Xavier Uniform initialization
        val = math.sqrt(6 / (prompt_dim + prompt_dim))  # Assuming fan_in and fan_out are both equal to prompt_dim

        # Initialize prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_layers, num_prompts, prompt_dim)) # Num_layers, Num_prompts, Prompt_dim
        nn.init.uniform_(self.prompt_embeddings, -val, val)  # Xavier Uniform initialization


    def incorporate_prompt(self, x, layer_num):
        """
        combine prompt embeddings with image-patch embeddings
        x: input embeddings for the block
        """
        B = x.shape[0] # number of mini-batch
        prompts = self.prompt_embeddings[layer_num,:,:].expand(B, -1, -1) # expands the prompts to the match the batch size

        if layer_num == 0: # if the input needs to go through the first block
            """
            the cat function outputs a tensor called x.
            x has a shape of (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
            """
            x = torch.cat((
                x[:, :1, :], # cls_token
                prompts,
                x[:, 1:, :] # patch_embeddings
            ), dim=1)
        else: # for the subsequent blocks
            # x[:, 1:1+self.num_prompts, :, :] = prompts
            x = torch.cat((
                x[:, :1, :], # cls_token
                prompts, # prompt_tokens
                x[:, (1+self.num_prompts):, :]
            ), dim=1)

        return x


class CustomViT(nn.Module):
    """
    This is the Custom ViT class for Visual Prompt Tuning!
    """
    def __init__(self, pretrained_model='vit_base_patch16_224',img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.): # get rid of unecessary parameters
        super().__init__()

        self.prompt_embeddings = CustomPrompts(num_prompts=50, prompt_dim=768, num_layers=12) # Create prompt embeddings
        # Define the pre-trained model with some modifications
        self.model = timm.create_model(pretrained_model, 
                          img_size=img_size, 
                          patch_size=patch_size, 
                          num_classes=num_classes, 
                          pretrained=True,
                          )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.model.forward_head(x)
        return x
    

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """
            This function overrides the existing function defined in the pre-trained model.
            """
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
            x = self.model.norm_pre(x)
            
            # Encoder = Blocks x 12
            for idx, block in enumerate(self.model.blocks): # 0~11
                x = self.prompt_embeddings.incorporate_prompt(x, idx) # Insert prompts in between the cls_token and patch embeddings
                x = block(x)

            x = self.model.norm(x) # Layer Normalization after the encoder
            return x