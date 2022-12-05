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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from collections import OrderedDict

from modules import PriorGrad
from dataset import AudioDataset
from utils import *

import soundfile as sf

class PGLearner:
    def __init__(self, hp, is_train):
        self.hp = hp
        self.dev = hp.DEV
        os.makedirs(hp.MODEL_DIR, exist_ok=True)
        os.makedirs(hp.RESULT_DIR, exist_ok=True)

        self.model = PriorGrad(hp).to(hp.DEV)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=hp.LR, weight_decay=hp.WEIGHT_DECAY)
        self.sche = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=hp.LR_DECAY, last_epoch=-1, verbose=True)
	

        train_ds = AudioDataset(
            hp.TRAIN_DIR if is_train else hp.VAL_DIR,
            hp=hp,
            train=True, 
            aug=hp.AUG
            )

        self.train_dl = DataLoader(
            train_ds,
            batch_size=hp.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=os.cpu_count()
            )

        val_ds = AudioDataset(
            hp.VAL_DIR if is_train else hp.TEST_DIR,
            hp=hp,
            train=False, 
            aug=False
            )
    
        self.val_dl = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=os.cpu_count()
            )

        self.autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=hp.AMP_SCALE)

        self.step = 0

        beta = np.array(hp.NOISE_SCHEDULE)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(hp.DEV)

        self.summary_writer = None
    
        fft_freq = torch.fft.fftfreq(hp.N_FFT, 1/hp.SR)
        self.lpf_idx = 0
        for f in fft_freq:
            if f>=hp.SPEECH_LPF or hp.USE_LPF_NOISE==False: break
            else:
                print(f'cutoff under {f}Hz at index:{self.lpf_idx}') 
                self.lpf_idx += 1

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.opt.state_dict().items()},
            'scheduler': self.sche.state_dict(),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['optimizer'])
        self.sche.load_state_dict(state_dict['scheduler'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.hp.MODEL_DIR}/{save_basename}'
        link_name = f'{self.hp.MODEL_DIR}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.hp.MODEL_DIR}/{filename}.pt')
            self.load_state_dict(checkpoint)
            print(f'succeed to restore model in {filename}.pt')
            return True
        except FileNotFoundError:
            print(f'weight file was not found')
            return False

    def randn_lpf(self, shape):
        win = torch.hann_window(self.hp.N_FFT).to(self.dev)
        noise = torch.randn(*shape).to(self.dev)
        spec = torch.stft(
            input=noise[...,:-1],
            n_fft=self.hp.N_FFT,
            hop_length=self.hp.HOP,
            win_length=self.hp.N_FFT,
            window=win,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)       
        
        spec[:, :self.lpf_idx] = 0

        spec = F.pad(spec, (0, 0, 0, 1),mode='reflect')

        rec = torch.istft(
            input=spec,
            n_fft=self.hp.N_FFT,
            hop_length=self.hp.HOP,
            win_length=self.hp.N_FFT,
            window=win,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        
        return rec

    def scaled_mse(self, out, target, target_std):
        # inverse of diagonal matrix is 1/x for each element
        sigma_inv = torch.reciprocal(target_std)
        mse_loss = (((out - target) * sigma_inv) ** 2)
        mse_loss = (mse_loss).sum() / torch.numel(out)
        return mse_loss

    def initial_gt_sample(self):
        with tqdm(self.val_dl) as pbar:
            for i, b in enumerate(pbar):
                if i<self.hp.NUM_SAMPLES:
                    aud = b['audio'].to(self.dev)
                    spec = b['spec'].to(self.dev)
                    tgt_std = b['target_std'].to(self.dev)       
                    a = aud.cpu().numpy()
                    save_spectrogram(
                        a[0], 
                        self.hp.RESULT_DIR,
                         f"_gt_{i}", 
                         self.hp.SR, 
                         self.hp.HOP
                         )
                    out = np.transpose(a, (1, 0))
                    sf.write(os.path.join(self.hp.RESULT_DIR, f"_gt_{i}.wav"), out, self.hp.SR)

                    lmspe = spec
                    
                    bs, _, w = lmspe.shape
                    if self.hp.USE_LPF_NOISE:
                        noise = self.randn_lpf((bs, w * self.hp.HOP)) * tgt_std
                    else:
                        noise = torch.randn(bs, 1, w * self.hp.HOP, device=self.hp.DEV) * tgt_std
                    noise = noise.cpu().squeeze(1).numpy()
                    save_spectrogram(
                        noise.reshape(-1),
                         self.hp.RESULT_DIR,
                          f"_input_{i}", 
                          self.hp.SR,
                          self.hp.HOP
                          )
                    out = np.transpose(noise, (1, 0))
                    if np.max(np.abs(out)) > 1.0:
                            out /= np.max(np.abs(out))
                    sf.write(os.path.join(self.hp.RESULT_DIR, f"_input_{i}.wav"), out, self.hp.SR)

                else: break

    def train_loop(self):
        self.model.train()
        while True:
            train_loss = 0
            with tqdm(self.train_dl, leave=False) as pbar:
                for b in pbar:
                    if self.step >= self.hp.N_ITER:
                        print('training done')
                        return

                    aud = b['audio'].to(self.dev)
                    spec = b['spec'].to(self.dev)
                    tgt_std = b['target_std'].to(self.dev)

                    loss = self.train_step(aud, spec, tgt_std)

                    pbar.set_postfix(OrderedDict(loss=f'{loss.item():.4f}'))
                    train_loss += loss.item() / len(self.train_dl)

                    if torch.isnan(loss).any():
                        raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                    if self.step > 0 and self.step % self.hp.LR_DECAY_STEP == 0:
                        self.sche.step()

                    if self.step % self.hp.SAMPLE_PREQ == 0:
                        self.valid_loop()

                    if self.step % self.hp.SAVE_FREQ == 0:
                        print("INFO: saving checkpoint at step {}".format(self.step))
                        self.save_to_checkpoint()
                    
                    self.step += 1

            print(f'Epoch:{self.step//len(self.train_dl):05d}, step:{self.step:07d}, train_loss:{train_loss:.4f}')
            self._write_summary(self.step, train_loss)

    def train_step(self, a, spec, tgt_std):
        for param in self.model.parameters():
            param.grad = None

        with self.autocast:
            t = torch.randint(0, len(self.hp.NOISE_SCHEDULE), [a.shape[0]], device=self.dev)
            ns = self.noise_level[t].unsqueeze(1)
            ns_sqrt = ns ** 0.5
            noise = self.randn_lpf(a.shape) if self.hp.USE_LPF_NOISE else torch.randn_like(a)
            noise = noise * tgt_std
            noisy = ns_sqrt * a + (1.0 - ns) ** 0.5 * noise

            pred_eps = self.model(noisy, spec, t)

            loss = self.scaled_mse(pred_eps.squeeze(1), noise, tgt_std)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.MAX_GRAD_NORM or 1e9)
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss

    def valid_loop(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            wav_val_loss = 0
            audio_preds = []
            with tqdm(self.val_dl, leave=False) as pbar:
                for i, b in enumerate(pbar):
                    aud = b['audio'].to(self.dev)
                    spec = b['spec'].to(self.dev)
                    tgt_std = b['target_std'].to(self.dev)

                    t = torch.randint(0, len(self.hp.NOISE_SCHEDULE), [aud.shape[0]], device=self.dev)
                    ns = self.noise_level[t].unsqueeze(1)
                    ns_sqrt = ns ** 0.5
                    noise = self.randn_lpf(aud.shape) if self.hp.USE_LPF_NOISE else torch.randn_like(aud)
                    noise = noise * tgt_std
                    noisy = ns_sqrt * aud + (1.0 - ns) ** 0.5 * noise

                    with self.autocast:
                        pred_eps = self.model(noisy, spec, t)
                    
                    loss = self.scaled_mse(pred_eps.squeeze(1), noise, tgt_std)

                    pbar.set_postfix(OrderedDict(loss=f'{loss.item():.4f}'))
                    val_loss += loss.item() / len(self.val_dl)

                    if i < self.hp.NUM_SAMPLES:
                        with self.autocast:
                            gen, state = self.sample(mspec=spec, target_std=tgt_std)
                        
                        loss = F.mse_loss(gen.squeeze(1), aud)
                        wav_val_loss += loss.item() / self.hp.NUM_SAMPLES

                        out = gen.cpu().numpy()        

                        audio_preds.append(out)
                             
                        gen = gen.squeeze().cpu().numpy()
                        
                        save_spectrogram(
                            gen.reshape(-1), 
                            self.hp.RESULT_DIR, 
                            f"{self.step:07d}_{i}_{state}", 
                            self.hp.SR, 
                            self.hp.HOP
                            )
                        out = np.transpose(out, (1, 0))
                        if np.max(np.abs(out)) > 1.0:
                            out /= np.max(np.abs(out))
                        sf.write(os.path.join(self.hp.RESULT_DIR, f"{self.step:07d}_{i}_{state}.wav"), out, self.hp.SR)
            
            print(f'Step:{self.step:07d}: val_loss:{val_loss:.4f}, sample_wav_loss:{wav_val_loss:.4f}')
            self._write_summary_valid(self.step, val_loss, wav_val_loss, audio_preds)
        self.model.train()
            
    def sample(self, mspec, target_std):
        if self.hp.USE_MANUAL_SCHE and torch.rand(1)>0.25:
            state = f'{len(self.hp.MANUAL_SCHE)}steps'
            sche = self.hp.MANUAL_SCHE
        else:
            state = f'{len(self.hp.NOISE_SCHEDULE)}steps'
            sche = self.hp.NOISE_SCHEDULE

        return self.predict(mspec, target_std, sche), state 

    def predict(self, spec, tgt_std, sche=None):
        with torch.no_grad():
            training_noise_schedule = np.array(self.hp.NOISE_SCHEDULE)
            if sche is None:
                inference_noise_schedule =  training_noise_schedule
            else:
                inference_noise_schedule = np.array(sche)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            if len(spec.shape) == 2:
                spec = spec.unsqueeze(0)
            spec = spec.to(self.dev)

            shape = (spec.shape[0], self.hp.HOP * spec.shape[-1])

            audio = self.randn_lpf(shape) if self.hp.USE_LPF_NOISE else torch.randn(*shape, device=self.dev)
            audio = audio * tgt_std

            for t in tqdm(range(len(alpha) - 1, -1, -1), leave=False):
                coef1 = 1 / alpha[t] ** 0.5
                coef2 = beta[t] / (1 - alpha_cum[t])**0.5
                audio = coef1 * (audio - coef2 * self.model(audio, spec, torch.tensor([T[t]], device=self.dev)).squeeze(1)) 
                if t > 0:
                    noise = self.randn_lpf(audio.shape) if self.hp.USE_LPF_NOISE else torch.randn_like(audio, device=self.dev)
                    noise = noise * tgt_std
                    sigma = ((1.0 - alpha_cum[t - 1]) / (1.0 - alpha_cum[t]) * beta[t]) ** 0.5
                    audio += sigma * noise
                if self.hp.USE_CLAMP:
                    audio = torch.clamp(audio, -1.0, 1.0)

        return audio

    def _write_summary(self, step, loss):
        writer = self.summary_writer or SummaryWriter(self.hp.MODEL_DIR, purge_step=step)
        writer.add_scalar('train/loss', loss, step)
        writer.flush()
        self.summary_writer = writer

    def _write_summary_valid(self, step, loss, wav_loss, audios):
        writer = self.summary_writer or SummaryWriter(self.hp.MODEL_DIR, purge_step=step)
        for i in range(len(audios)):
            writer.add_audio(f'valid/audio_pred_{i:02d}', audios[i], step, sample_rate=self.hp.SR)
        writer.add_scalar('valid/loss', loss, step)
        writer.add_scalar('valid/wav_loss', wav_loss, step)
        writer.flush()
        self.summary_writer = writer
