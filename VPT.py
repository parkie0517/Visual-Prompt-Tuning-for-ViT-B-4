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
    def __init__(self, num_layers, num_prompts, embed_dim):
        super(Prompt, self).__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.prompt_embeddings = nn.ModuleList(
            [nn.Parameter(torch.randn(num_prompts, embed_dim)) for _ in range(num_layers)]
        )
        self.initialize_embeddings()


    def initialize_embeddings(self):
        """
        Performs Xavier initialization to the prompt embeddings
        """
        for embedding in self.prompt_embeddings:
            init.xavier_uniform_(embedding)  # Xavier uniform initialization


    def forward(self, x, layer_idx):
        if layer_idx >= self.num_layers:
            return x
        B, N, C = x.shape
        prompt_embeddings = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prompt_embeddings, x], dim=1)