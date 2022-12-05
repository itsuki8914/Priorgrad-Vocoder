import os
import numpy  as np
import torch 
import librosa
import librosa.display 
import matplotlib.pyplot as plt

def torch_fix_seed(seed=42):
    # Python random
    #random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = True

def save_spectrogram(y, dirname, name, sr=24000, hop=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(S_DB, y_axis='log', sr=sr, x_axis='time')
    plt.colorbar()
    plt.savefig(f"{dirname}/{name}.jpg", dpi=300)
    plt.close()

def makedirs(name):
    os.makedirs(name, exist_ok=True)
