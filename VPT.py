"""
VPT.py

This file has the VPT class.
Import this file to use the VPT module!
"""


# 1. Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.init as init


class Prompt(nn.Module):
    def __init__(self, num_layers=12, num_prompts=50, embed_dim=768):
        super(Prompt, self).__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        self.prompt_embeddings = nn.Parameter(torch.randn(num_layers, num_prompts, embed_dim))
        
        self.initialize_embeddings()



    def initialize_embeddings(self):
        """
        Performs Xavier initialization to the prompt embeddings
        """
        torch.nn.init.xavier_uniform_(self.prompt_embeddings)

    
    def forward(self, x, layer_idx):
        if layer_idx >= self.num_layers:
            return x
        B, N, C = x.shape
        cls_token, patches = x[:, :1, :], x[:, 1:, :]  # Separate cls_token and patches
        prompt_embeddings = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(B, -1, -1)
        return torch.cat([cls_token, prompt_embeddings, patches], dim=1) # cls_token, prompts, tokens