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

from utils import *
from learner import PGLearner
from hparams import HPARAMS
from argparse import ArgumentParser

import os, glob
import numpy as np
import soundfile as sf

import torch
import torchaudio as ta
import torchaudio.transforms as tat
import torch.nn.functional as F

def save_wav_spec(hp, fname, aud, state):
    basename = os.path.basename(fname).split('.')[0]
    out = aud.cpu().numpy()                 
    gen = aud.squeeze().cpu().numpy()
    save_spectrogram(
        gen.reshape(-1), 
        hp.RESULT_DIR, 
        f"{basename}_{state}", 
        hp.SR, 
        hp.HOP
        )
    out = np.transpose(out, (1, 0))
    if np.max(np.abs(out))>1:
        out /= np.maximum(0.01, np.max(np.abs(out)))
    sf.write(os.path.join(hp.RESULT_DIR, f"{basename}_{state}.wav"), out, hp.SR)

def aud2aud(trainer, hp):
    audio_files = glob.glob(f'{hp.TEST_DIR}/**/*.wav', recursive=True)

    win = torch.hann_window(hp.N_FFT)
    fb = tat.MelScale(
            n_mels=hp.N_MELS,
            sample_rate=hp.SR,
            f_min=hp.F_MIN,
            f_max=hp.F_MAX,
            n_stft=hp.N_FFT//2 + 1,
            norm='slaney',
            mel_scale='slaney'
        )

    for f in audio_files:
        a, sr = ta.load(f)
        if hp.AUDIO_NORM:
            a /= torch.maximum(torch.tensor(0.01), torch.max(torch.abs(a)))
        if a.shape[0]>1:
            a = a.mean(0).unsqueeze(0)
        if sr != hp.SR:
            print(f'warning, loading file{f} has sr:{sr} and set:{hp.SR} is different')
            resampler = tat.Resample(sr, hp.SR)
            a = resampler(a)
        p = (a.shape[-1] // hp.HOP + 1) * hp.HOP - a.shape[-1]
        a = F.pad(a, (0, p), mode='constant')

        p = (hp.N_FFT - hp.HOP) // 2
        
        y = F.pad(a, [p, p], mode='reflect')
        spec = torch.stft(
            input=y,
            n_fft=hp.N_FFT,
            hop_length=hp.HOP,
            win_length=hp.N_FFT,
            window=win,
            center=False,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        mspec = fb(spec)
        lmspec = torch.log(torch.clamp(mspec, min=1e-5))
        energy = mspec.sum(1).sqrt()
        tgt_std = torch.clamp(energy / hp.ENERGY_MAX, min=hp.STD_MIN, max=None)
        tgt_std = torch.repeat_interleave(tgt_std, hp.HOP).to(hp.DEV)
        
        lmspec = lmspec.to(hp.DEV)

        if hp.USE_MANUAL_SCHE:
            state = f'{len(hp.MANUAL_SCHE)}steps'
            sche = hp.MANUAL_SCHE
        else:
            state = f'{len(hp.NOISE_SCHEDULE)}steps'
            sche = hp.NOISE_SCHEDULE

        with trainer.autocast:
            rec_aud = trainer.predict(lmspec, tgt_std, sche)
        
        save_wav_spec(hp, f, rec_aud, state)

def mel2aud(trainer, hp):
    mel_files = glob.glob(f'{hp.TEST_DIR}/**/*.npy', recursive=True)
    for f in mel_files:        
        lmspec = np.load(f)
        energy = lmspec.exp().sum(1).sqrt()
        tgt_std = torch.clamp(energy / hp.ENERGY_MAX, min=hp.STD_MIN, max=None)
        tgt_std = torch.repeat_interleave(tgt_std, hp.HOP).to(hp.DEV) 

        lmspec = torch.from_numpy(lmspec).to(hp.DEV) 

        if hp.USE_MANUAL_SCHE:
            state = f'{len(hp.MANUAL_SCHE)}steps'
            sche = hp.MANUAL_SCHE
        else:
            state = f'{len(hp.NOISE_SCHEDULE)}steps'
            sche = hp.NOISE_SCHEDULE

        with trainer.autocast:
            rec_aud = trainer.predict(lmspec, tgt_std, sche)
        
        save_wav_spec(hp, f, rec_aud, state)


if __name__ == '__main__':
    parser = ArgumentParser(description='train a WaveGrad model')
    parser.add_argument('--mode','-m',default='base', help='model_scale')
    args = parser.parse_args()
    
    hp = HPARAMS(args.mode)
    torch_fix_seed(hp.SEED)
    trainer = PGLearner(hp, False)
    trainer.restore_from_checkpoint()

    if hp.direction=='a2a':
        with torch.no_grad():
            aud2aud(trainer, hp)
    else:
        with torch.no_grad():
            mel2aud(trainer, hp)
    
