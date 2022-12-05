# Priorgrad-Vocoder
Unofficial implementation of PriorGrad

### This Repository is inspired below codes, See also them.
### [Official implementation of PriorGrad](https://github.com/microsoft/NeuralSpeech)
### [lmnt-com/diffwave](https://github.com/lmnt-com/diffwave)

# Original-Paper

[PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior](https://arxiv.org/abs/2106.06406)

# Usage

1. Put the folder containing the wav files for training in named datasets.

   And Put the folder containing a few wav files for validation in datasets_val.

like this

```
...
│
train
|   │
|   ├── speaker_1
|   │     ├── wav1_1.wav
|   │     ├── wav1_2.wav
|   │     ├── ...
|   │     └── wav1_i.wav
|   ├── speaker_2
|   │     ├── wav2_1.wav
|   │     ├── wav2_2.wav
|   │     ├── ...
|   │     └── wav2_j.wav 
|   ...
|   └── speaker_N
|         ├── wavN_1.wav
|         ├── wavN_2.wav
|         ├── ...
|         └── wavN_k.wav    
valid
|   │
|   ├── valid_wavs
|   │     ├── wavM_1.wav
|   │     ├── wavM_2.wav
|   │     ├── ...
|   │     └── wavM_l.wav
|   |
...
├── run.py     
├── hparams.py
...
```

2. Adjust training Hyperparameter. Open hparams.py and change the values as you want to train.

3. Train PriorGrad to run run.py.

   if you want to train custom model, clarify with --mode xxx. base model is trained as default.

example (When you want to train large model)

```
python3 run.py --mode large
```

4. After Train, You can run inference.py for the test data.

   Match the value of TEST_DIR in hparams.py with the name of the folder you want to infer.
   
   Please select one MANUAL_SCHE. Inference is faster and lower quality in order from the top.
   
example (when you want to select 6time diffusion process)
   
```
#self.MANUAL_SCHE = [1e-2, 1e-1, 0.68611391673] # Manual-3
self.MANUAL_SCHE = [1e-4, 1e-3, 1e-2, 5e-2, 2e-1, 5e-1] # PriorGrad-6
#self.MANUAL_SCHE = [1e-4, 5e-4, 8e-4, 1e-3, 5e-3, 8e-3, 1e-2, 5e-2, 8e-2, 0.1, 0.2, 0.5] # PriorGrad-12
#self.MANUAL_SCHE = np.linspace(1e-4, 0.05, 50).tolist() # PriorGrad-50
```

   then run inference.py, do not forget put --mode (trained model scale) at end.
   
example (When you trained small model)

```
python3 inference.py --mode small
```

# Result Examples

You can listen to the result of training with this source code.

In [result_examples_500K](https://github.com/itsuki8914/Priorgrad-Vocoder/tree/main/result_examples_500K), it contains the results of a model trained 500000 steps.

[JVS (Japanese versatile speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus?authuser=0) is Used for Training Dataset.

and [Voice Actress Statical Corpus(声優統計コーパス)](https://voice-statistics.github.io/) was used for validation and test datasets.
