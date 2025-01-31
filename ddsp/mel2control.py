
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from ddsp.model_conformer_naive import ConformerNaiveEncoder


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Mel2Control(nn.Module):
    def __init__(
            self,
            n_mels,
            n_sin_hars,
            block_size,
            output_splits
        ):
        super().__init__()
        self.output_splits = output_splits        
        self.mel_emb = nn.Linear(n_mels, 256)
        self.phase_emb = nn.Linear(n_sin_hars, 256)
        self.decoder = ConformerNaiveEncoder(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            use_norm=False,
            conv_only=True,
            conv_dropout=0,
            atten_dropout=0.1)
        self.norm = nn.LayerNorm(256)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def forward(self, mel, inp):
        
        '''
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        '''
        # print("mel2control mel",mel.shape)
        x = self.mel_emb(mel) + self.phase_emb(inp)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls 

