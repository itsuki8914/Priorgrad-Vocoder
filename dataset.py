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

import numpy as np
import os
import random
import torch
import torchaudio as ta
import torchaudio.transforms as tat
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob

class AudioDataset(Dataset):
    def __init__(
        self,
        root,
        hp,
        train=True,
        aug=False
    ):
        super().__init__()
        self.hp = hp
        self.sr = hp.SR
        self.hop = hp.HOP
        self.train = train
        self.aug = aug
        self.wav_len = hp.HOP * hp.CROP_MEL_FRAMES 

        self.resampler = None
        
        self.files = glob.glob(f'{root}/**/*.wav', recursive=True)
        self.win_fn, self.mel_fb = self.create_win_fb(hp)

    def create_win_fb(self, hp):
        win = torch.hann_window(hp.N_FFT)
        # DO NOT CHANGE norm AND mel_scale FROM slaney
        fb = tat.MelScale(
            n_mels=hp.N_MELS,
            sample_rate=hp.SR,
            f_min=hp.F_MIN,
            f_max=hp.F_MAX,
            n_stft=hp.N_FFT//2 + 1,
            norm='slaney',
            mel_scale='slaney'
        )
        return win, fb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        a, sr = ta.load(self.files[i])
        a /= torch.maximum(torch.tensor(0.01), torch.max(torch.abs(a)))

        if a.shape[0]>1:
            # has 2 or more channels
            a = a.mean(0).unsqueeze(0)

        if sr != self.sr:
            print(f'warning, loading file{self.files[i]} has sr:{sr} and set:{self.sr} is different')
            self.resampler = tat.Resample(sr, self.sr)
            a = self.resampler(a)
        
        if not self.train:
            p = (a.shape[-1] // self.hop + 1) * self.hop - a.shape[-1]
            a = F.pad(a, (0, p), mode='constant').data

        elif a.shape[-1]>self.wav_len:
            st = torch.randint(low=0,high=a.size(-1)-self.wav_len,size=(1,))[0]
            a = a[...,st:st+self.wav_len]
        
        else:
            p = self.wav_len - a.shape[-1]
            a = F.pad(a, (0, p), mode='constant').data

        if self.aug:
            if torch.rand(1)>0.2:
                amp = torch.rand(1)*0.7 + 0.3
                a *= amp

        p = (self.hp.N_FFT - self.hp.HOP) // 2
        
        y = F.pad(a, [p, p], mode='reflect')
        spec = torch.stft(
            input=y,
            n_fft=self.hp.N_FFT,
            hop_length=self.hp.HOP,
            win_length=self.hp.N_FFT,
            window=self.win_fn,
            center=False,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        mspec = self.mel_fb(spec)
        lmspec = torch.log(torch.clamp(mspec, min=1e-5))
        energy = mspec.sum(1).sqrt()
        tgt_std = torch.clamp(energy / self.hp.ENERGY_MAX, min=self.hp.STD_MIN, max=self.hp.ENERGY_MAX)
        tgt_std = torch.repeat_interleave(tgt_std, self.hp.HOP)

        return {
                'audio':a.squeeze(),
                'spec':lmspec[0],
                'target_std':tgt_std
                }
