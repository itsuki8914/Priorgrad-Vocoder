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
import torch

class HPARAMS:
    def __init__(self, mode='base'):
        # common settings
        self.DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # path settings
        self.MODEL_DIR = 'model'
        self.RESULT_DIR = 'result'
        self.TRAIN_DIR = 'train'
        self.VAL_DIR = 'valid'

        # difffusion settings
        self.NOISE_SCHEDULE = np.linspace(1e-4, 0.05, 50).tolist()
        self.USE_LPF_NOISE = True
        self.SPEECH_LPF = 70 # Hz

        # train settings
        self.SEED = 8620
        self.RESUME = False
        self.AUG = True
        self.BATCH_SIZE = 32
        self.LR = 2e-4 / 32 * self.BATCH_SIZE 
        self.WEIGHT_DECAY = 1e-8
        self.LR_DECAY = 0.998
        self.LR_DECAY_STEP = 1000 * 32 // self.BATCH_SIZE 
        self.N_ITER = 1000000
        self.SAMPLE_PREQ = 10000
        self.SAVE_FREQ = 50000
        self.NUM_SAMPLES = 7
        self.MAX_GRAD_NORM = None
        self.USE_AMP = True
        self.AMP_SCALE = 2048 # to avoid NaN, make smaller this param, but when too small, vanishing grad

        # validation and inference settings
        self.direction = 'a2a' # audio2audio
        #self.direction = 'm2a' # log-melspec2audio
        self.TEST_DIR = 'valid'
        self.AUDIO_NORM = False
        self.USE_CLAMP = False
        self.USE_MANUAL_SCHE = True
        self.MANUAL_SCHE = [1e-2, 1e-1, 0.68611391673] # Manual-3
        #self.MANUAL_SCHE = [1e-4, 1e-3, 1e-2, 5e-2, 2e-1, 5e-1] # PriorGrad-6
        #self.MANUAL_SCHE = [1e-4, 5e-4, 8e-4, 1e-3, 5e-3, 8e-3, 1e-2, 5e-2, 8e-2, 0.1, 0.2, 0.5] # PriorGrad-12
        #self.MANUAL_SCHE = np.linspace(1e-4, 0.05, 50).tolist() # PriorGrad-50 (same at USE_MANUAL_SCHE==False)
        
        # audio settings
        self.SR = 24000
        self.N_FFT = 1024
        self.HOP = 256
        self.F_MIN = 20
        self.F_MAX = self.SR / 2 
        self.STD_MIN = 0.1
        self.ENERGY_MAX = 5.0

        # model scale
        if mode == 'tiny' or mode == 't':
            # 0.4M params
            self.N_MELS = 64
            self.CROP_MEL_FRAMES = 20
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 32
            self.EMB_IN = 32
            self.EMB_OUT = 128
            self.SPEC_US_FACTORS = [16, 16]
            self.SPEC_US_HIDDEN = 1
            self.DILARION_CYCLE = 8
            self.N_RES = 24
            self.PROJ_KS = 1

        elif mode == 'small' or mode == 's':
            # 1.1M params
            self.N_MELS = 80
            self.CROP_MEL_FRAMES = 24
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 48
            self.EMB_IN = 48
            self.EMB_OUT = 256
            self.SPEC_US_FACTORS = [16, 16]
            self.SPEC_US_HIDDEN = 32
            self.DILARION_CYCLE = 9
            self.N_RES = 27
            self.PROJ_KS = 1

        elif mode == 'base' or mode == 'b':
            # 3.7M params
            self.N_MELS = 80
            self.CROP_MEL_FRAMES = 32
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 64
            self.EMB_IN = 64
            self.EMB_OUT = 512
            self.SPEC_US_FACTORS = [16, 16]
            self.SPEC_US_HIDDEN = 64
            self.DILARION_CYCLE = 10
            self.N_RES = 30
            self.PROJ_KS = 3

        elif mode == 'large' or mode == 'l':
            #13.6M params
            self.N_MELS = 128
            self.CROP_MEL_FRAMES = 48
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 96
            self.EMB_IN = 96
            self.EMB_OUT = 768
            self.SPEC_US_FACTORS = [4, 4, 4, 4]
            self.SPEC_US_HIDDEN = 96
            self.DILARION_CYCLE = 12
            self.N_RES = 36
            self.PROJ_KS = 5
    
        elif mode == 'xlarge' or mode == 'xl':
            # 35.5M params
            self.N_MELS = 128
            self.CROP_MEL_FRAMES = 64
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 128
            self.EMB_IN = 128
            self.EMB_OUT = 1024
            self.SPEC_US_FACTORS = [4, 4, 4, 2, 2]
            self.SPEC_US_HIDDEN = 128
            self.DILARION_CYCLE = 12
            self.N_RES = 48
            self.PROJ_KS = 7
        
        elif mode == 'paper' or mode == 'p':
            # like official implementation
            # 2.6M params
            self.N_MELS = 80
            self.CROP_MEL_FRAMES = 62
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 64
            self.EMB_IN = 64
            self.EMB_OUT = 512
            self.SPEC_US_FACTORS = [16, 16]
            self.SPEC_US_HIDDEN = 1 
            self.DILARION_CYCLE = 10
            self.N_RES = 30
            self.PROJ_KS = 1

        else:
            # this may be your cunstom setting
            self.N_MELS = 80
            self.CROP_MEL_FRAMES = 24
            self.WAV_LEN = self.HOP * self.CROP_MEL_FRAMES
            self.RES_CHANNELS = 96
            self.EMB_IN = 64
            self.EMB_OUT = 768
            self.SPEC_US_FACTORS = [4, 4, 4, 4]
            self.SPEC_US_HIDDEN = 96
            self.DILARION_CYCLE = 12
            self.N_RES = 36
            self.PROJ_KS = 5


        assert np.prod(self.SPEC_US_FACTORS)==self.HOP, 'prod(us_factors) must match hop_len'


