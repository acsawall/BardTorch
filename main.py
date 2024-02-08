import torch
import torchaudio
from torchaudio.transforms import FFTConvolve, Spectrogram, Resample, TimeStretch, TimeMasking, FrequencyMasking
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import wave
from maad import sound, rois, features
import os
import sounddevice as sd
import time
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa

FSC22_labels_map = {1: "Fire",
                    2: "Rain",
                    3: "Thunderstorm",
                    4: "WaterDrops",
                    5: "Wind",
                    6: "Silence",
                    7: "TreeFalling",
                    10: "Axe",
                    13: "Handsaw",
                    14: "Firework",
                    16: "WoodChop",
                    17: "Whistling",
                    18: "Speaking",
                    19: "Footsteps",
                    20: "Clapping",
                    21: "Insect",
                    22: "Frog",
                    23: "BirdChirping",
                    24: "WingFlapping",
                    26: "WolfHowl",
                    27: "Squirrel"
                    }


# Borrowed from https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials
# -audio-feature-extractions-tutorial-py
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


# Borrowed from https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials
# -audio-feature-extractions-tutorial-py
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

class AudioPipeline(torch.nn.Module):
    def __init__(
            self,
            input_freq=16000,
            resample_freq=8000,
            n_fft=1024,
            stretch_factor=0.8):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)
        # Real-valued spectrogram
        self.spec = Spectrogram(n_fft=n_fft, power=2)
        #self.spec_aug = torch.nn.Sequential(
        #    TimeStretch(stretch_factor, fixed_rate=True),
        #    FrequencyMasking(freq_mask_param=80),
        #    TimeMasking(time_mask_param=80)
        #)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        spec = self.spec(resampled)
        #spec = self.spec_aug(spec)
        return spec


class AudioDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir

    def __len__(self):
        return len(os.listdir(self.audio_dir))

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, os.listdir(self.audio_dir)[idx])
        audio = torchaudio.load(audio_path)
        return audio


# Play audio given signal x time data and a sample rate
def play(data, sr):
    dur = len(data) / sr
    sd.play(data, sr)
    time.sleep(dur)
    sd.stop()


# Load dataset path
def load_files(path):
    if os.path.exists(path):
        counter = 0
        for filename in os.listdir(path):
            if counter >= 5:
                break
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                print("Found", filename)
                s, sr = sf.read(filepath)
                f, t, stft = signal.stft(s, sr, nperseg=1024)
                plt.pcolormesh(t, f, np.abs(stft))
                plt.show()
                time.sleep(.5)  # prevent timeouts from matplotlib
                counter += 1


def train_loop(dataloader, model):
    size = len(dataloader.dataset)
    model.train()

    count = 0
    for batch, (data, sr) in enumerate(dataloader):
        features = model(data.to(device))
        f = features.cpu().data.numpy().argmax()
        print(f)


def test_loop(dataloader, model, loss_fn):
    # TODO Use FSC22 Dataset as test data
    pass


if __name__ == '__main__':
    fire_path = "D:/datasets/ESC-50/203 - Crackling fire"
    # load_files(fire_path)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device == "cuda":
        print(f"Running on {torch.cuda.get_device_name(device)}")

    pipeline = AudioPipeline()
    pipeline.to(device)
    print(pipeline)

    data, sr = torchaudio.load("D:/datasets/ESC-50/203 - Crackling fire/1-4211-A.ogg")
    '''spectrogram = Spectrogram(n_fft=512)
    spec = spectrogram(data)

    fig, axs = plt.subplots(2, 1)
    plot_waveform(data, sr, ax=axs[0])
    plot_spectrogram(spec[0], ax=axs[1])
    #print(spec[0])
    fig.tight_layout()
    plt.show()'''

    for name, param in pipeline.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    training_data = AudioDataset("D:/datasets/ESC-50/203 - Crackling fire")
    train_dataloader = DataLoader(training_data, shuffle=True)
    #features = pipeline(data.to(device))
    #print(features)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, pipeline)
    print("Done!")

