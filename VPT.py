import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class CustomEmbeddingLayer(nn.Module):
    """
    Custom Embedding Layer includes the following thigns.
        1. Embedding layer (Uses Conv2D operation for efficiency)
        2. Position embedding layer
    """
    def __init__(self, in_chans, embed_dim, img_size, patch_size, trainable_pos_embed=True):
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Using Conv2d operation to perform linear projection

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Create a classification token (used later for classifying the image)
        self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim)) # Create positional embedding parameters

        nn.init.normal_(self.cls_token, std=1e-6) # Initialize classfication token using normal(Gaussian) distibution
        trunc_normal_(self.pos_embed, std=.02) # Initialize positional embeddings using truncated normal distribution

        if trainable_pos_embed:
            self.pos_embed.requires_grad = True # Allow positional embeddings to be trainable
        else:
            self.pos_embed.requires_grad = False # Do not allow positional embeddings to be trainable

    def forward(self, x): # Define the forward function
        B, C, H, W = x.shape
        embedding = self.project(x) # Perform Linear projection (=tokenization of the image)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

        # Add the classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)

        # Add the position embedding
        z = z + self.pos_embed
        return z
    

class CustomViT(nn.Module):
    def __init__(self, vit_model, in_chans=3, img_size=32, patch_size=4, num_classes=10, num_prompts=50, prompt_dim=768):
        super(CustomViT, self).__init__()
        self.model = vit_model

        self.embedding_layer = CustomEmbeddingLayer(in_chans, prompt_dim, img_size, patch_size)
        
        # Initialize prompts
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        # Replace MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.embedding_layer(x) # Patch, position embedding



        # Append prompts
        prompts = self.prompts.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat((prompts, x), dim=1)

        # Pass through the ViT model
        x = self.model.forward_features(x)

        # Apply MLP head
        x = self.mlp_head(x[:, 0])

        return x

