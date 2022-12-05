# Copyright 2022 (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

class Emb(nn.Module):
    def __init__(self, n_steps=50, in_c=64, out_c=512):
        super().__init__()
        #self.emb = self.build_emb(n_steps, in_c)
        self.register_buffer('emb', self.build_emb(n_steps, in_c), persistent=False)
        self.proj1 = nn.Linear(in_c*2, out_c)
        self.act1 = nn.PReLU()
        self.proj2 = nn.Linear(out_c, out_c)
        self.act2 = nn.PReLU()

    def build_emb(self, n_steps, in_c):
        steps = torch.arange(n_steps).unsqueeze(1) # [T, 1]
        dims = torch.arange(in_c).unsqueeze(0) # [1, in_c]
        table = steps * 10 ** (dims * 4.0 /(in_c - 1)) # [T, in_c]
        table = torch.cat([table.sin(), table.cos()], dim=1)
        return table
    
    def lerp_emb(self, t):
        li = torch.floor(t).long()
        hi = torch.ceil(t).long()
        low = self.emb[li]
        high = self.emb[hi]
        return low + (high - low) * (t - li.float()).unsqueeze(1)
    
    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.emb[t]
        else:
            x = self.lerp_emb(t)

        x = self.proj1(x)
        x = self.act1(x)
        x = self.proj2(x)
        return self.act2(x)

class SpecUS(nn.Module):
    def __init__(self, us_factors=[16, 16], hidden=1):
        super().__init__()
        for f in us_factors:
            assert f%2==0, 'us factors must be evens all'

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(1, hidden, [3, us_factors[0]*2], stride=[1, us_factors[0]], padding=[1, us_factors[0]//2]),
            nn.PReLU()
        ])
        for f in us_factors[1:-1]:
            self.layers.append(nn.ConvTranspose2d(hidden, hidden, [3, f*2], stride=[1, f], padding=[1, f//2]))
            self.layers.append(nn.PReLU())

        self.layers.append(nn.ConvTranspose2d(hidden, 1, [3, us_factors[-1]*2], stride=[1, us_factors[-1]], padding=[1, us_factors[-1]//2]))
        self.layers.append(nn.PReLU())

    def forward(self, x):
        x = x.unsqueeze(1)
        for l in self.layers:
            x = l(x)
        return x.squeeze(1)

class ResBlk(nn.Module):
    def __init__(self, n_mels, res_c, dilation, diff_proj_dim=512, proj_ks=3):
        super().__init__()
        self.dilated_conv = nn.Conv1d(res_c, res_c*2, 3, padding=dilation, dilation=dilation)
        self.diff_proj = nn.Linear(diff_proj_dim, res_c)
        self.cond_proj = nn.Conv1d(n_mels, res_c*2, proj_ks, padding=proj_ks//2)
        self.out_proj = nn.Conv1d(res_c, res_c*2, proj_ks, padding=proj_ks//2)

    def forward(self, x, cond, diff_step):
        diff_step = self.diff_proj(diff_step).unsqueeze(-1)
        cond = self.cond_proj(cond)

        y = x + diff_step
        y = self.dilated_conv(y) + cond

        g, f = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(g) * torch.tanh(f)
        
        y = self.out_proj(y)
        res, skip = torch.chunk(y, 2, dim=1)
        return (x + res) / sqrt(2), skip
           
class PriorGrad(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.inp_proj = nn.Conv1d(1, hp.RES_CHANNELS, hp.PROJ_KS, padding=hp.PROJ_KS//2)
        self.act1 = nn.PReLU()
        self.diffusion_emb = Emb(len(hp.NOISE_SCHEDULE), hp.EMB_IN, hp.EMB_OUT)
        self.spec_us = SpecUS(hp.SPEC_US_FACTORS, hp.SPEC_US_HIDDEN)

        self.resblks = nn.ModuleList([
            ResBlk(hp.N_MELS, hp.RES_CHANNELS, 2**(i % hp.DILARION_CYCLE), hp.EMB_OUT, hp.PROJ_KS)
            for i in range(hp.N_RES)
        ])

        self.skip_proj = nn.Conv1d(hp.RES_CHANNELS, hp.RES_CHANNELS, hp.PROJ_KS, padding=hp.PROJ_KS//2)
        self.act2 = nn.PReLU()
        self.out_proj = nn.Conv1d(hp.RES_CHANNELS, 1, hp.PROJ_KS, padding=hp.PROJ_KS//2)

        print('num params: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, a, spec, t):
        x = a.unsqueeze(1)
        x = self.inp_proj(x)
        x = self.act1(x)

        diff_step = self.diffusion_emb(t)
        spec = self.spec_us(spec)

        skip = []
        for l in self.resblks:
            x, s = l(x, spec, diff_step)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.resblks))
        x = self.skip_proj(x)
        x = self.act2(x)
        return self.out_proj(x)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from hparams import HPARAMS
    parser = ArgumentParser(description='train (or resume training) a PriorGrad model')
    parser.add_argument('--mode','-m',default='base', help='model_size')
    args = parser.parse_args()
    hp = HPARAMS(args.mode)
    net = PriorGrad(hp)
    aud = torch.randn(3, hp.WAV_LEN)
    mspec = torch.randn(3, hp.N_MELS, hp.CROP_MEL_FRAMES)
    nl = torch.tensor([i for i in range(3)], dtype=torch.float)
    print(f'input_shape, audio:{aud.numpy().shape}, melspec:{mspec.numpy().shape}, noise_level:{nl.numpy().shape}')
    out = net(aud, mspec, nl)
    print(f'output_shape: {out.detach().numpy().shape}')