"""
VPT.py

This file has the VPT class.
Import this file to use the VPT module!
"""

import torch

class VisualPrompt(nn.Module):
    def __init__(self, num_prompts, embed_dim):
        super(VisualPrompt, self).__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, embed_dim))

    def forward(self, x):
        B, N, C = x.shape
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prompt_embeddings, x], dim=1)
