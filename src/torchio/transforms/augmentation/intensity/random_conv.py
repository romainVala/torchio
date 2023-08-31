from collections import defaultdict
from typing import Dict
from typing import Tuple

import numpy as np
import torch

from .. import RandomTransform
from ... import IntensityTransform
from ....data.subject import Subject
from ....typing import TypeRangeFloat
from ....utils import to_tuple

from torch import nn
from torch.nn import functional as F


class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, init_scale = 'default', **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        self.init_scale     = init_scale
        assert requires_grad == False

    def forward(self, x_in, requires_grad = False):
        # random size of kernel
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]
        print(f'k is {k}')

        nb, nc, nx, ny, nz = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k, k  ], requires_grad = self.requires_grad  )
        shift = torch.randn( [self.out_channel * nb, 1, 1, 1 ], requires_grad = self.requires_grad  ) * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        x_conv = F.conv3d(x_in, ker, stride =1, padding = k // 2, dilation = 1, groups = nb )
        #romain test
        #padd = 1 if k==3 else 0
        #x_conv = F.conv3d(x_in, ker, stride=1, padding=padd)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv

class GINGroupConv3D(nn.Module):
    def __init__(self, out_channel = 1, in_channel = 1, interm_channel = 1, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', init_scale = 'default', **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv3D, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = 0)
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale,layer_id = ii + 1)
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = n_layer - 1, use_act = False)
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        nb, nc, nx, ny, nz = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None, None] # nb, 1, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1, 1) # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        #mixed = alphas * x + (1.0 - alphas) * x_in
        mixed = x

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.contiguous().view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed

###### unit test #####
'''
if __name__ == '__main__':
    from pdb import set_trace
    xin = torch.rand([5, 3, 64, 64, 32]).cuda()
    augmenter = GINGroupConv3D().cuda()
    out = augmenter(xin)
    set_trace()
    print(out.shape)


'''

class RandomConv(RandomTransform, IntensityTransform):
    r"""Randomly change contrast of an image by passing through a small convolution block with random weights

    """  # noqa: B950

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:

        Gin_conv = GINGroupConv3D()

        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = Gin_conv(image.data.unsqueeze(0).float())
            image.set_data(transformed_tensors[0])

        return subject

