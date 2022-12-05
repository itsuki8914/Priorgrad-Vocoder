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

if __name__ == '__main__':

    parser = ArgumentParser(description='train a PriorGrad model')
    parser.add_argument('--mode','-m',default='base', help='model_scale')
    args = parser.parse_args()
    
    hp = HPARAMS(args.mode)
    torch_fix_seed(hp.SEED)
    makedirs(hp.RESULT_DIR)
    makedirs(hp.MODEL_DIR)
    trainer = PGLearner(hp, True)
    if hp.RESUME:
        trainer.restore_from_checkpoint()
    trainer.initial_gt_sample()
    trainer.train_loop()
