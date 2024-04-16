import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
import torchaudio.io
import torchvision
from torchvision.transforms import ToPILImage
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageShow


# Universal ESCDataset file for processing as waveforms, mel spectrogram's, or images
class ESCDataset(Dataset):
    """
    :param audio_dir: str, path to dataset location on disk
    :param target_sample_rate: int, sample rate to resampling
    :param num_samples: int, number of samples to use as a standardize length for trimming or padding
    :param device:
    :param ret_type: str, one of ["waveform", "mel", "image"] for either a waveform, mel spectrogram, or PIL image
    """
    def __init__(self, audio_dir, target_sample_rate, num_samples, device, ret_type: str = "waveform"):
        assert ret_type in ["waveform", "mel", "image"], \
            "Parameter \"ret_type\" must be one of [\"waveform\", \"mel\", \"image\"]"
        self.audio_dir = audio_dir
        annotations = pd.DataFrame(columns=["folder", "file"])
        for folder in os.listdir(audio_dir):
            if os.path.isdir(audio_dir + "/" + folder):
                for file in os.listdir(audio_dir + "/" + folder):
                    annotations = pd.concat([annotations, pd.DataFrame({"folder": [folder], "file": [file]})], ignore_index=True)
        self.annotations = annotations
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.ret_type = ret_type

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Load audio from disk
        signal, sr = torchaudio.load(audio_sample_path)

        # Apply waveform transforms
        signal = self._resample(signal, sr)
        signal = self._mix_to_mono(signal)
        signal = self._resize_signal_tensor(signal)
        signal = signal.to(self.device)
        if self.ret_type == "waveform":
            return signal, label

        # Convert to mel spectrogram
        mel_transform = MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=1024 // 2,   # 4,
            n_mels=128
        ).to(self.device)
        mel_spec = mel_transform(signal)

        # Scale logarithmically
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        if self.ret_type == "mel":
            return mel_spec

        # Convert to PIL image
        img = self._mel_to_img(mel_spec)
        return img, label

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_to_mono(self, signal):
        # signal.shape -> (channels, samples) -> (1, samples)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resize_signal_tensor(self, signal):
        len_signal = signal.shape[1]
        # Trim signal if too long
        if len_signal > self.num_samples:
            signal = signal[:, :self.num_samples]
        # Pad signal if too short
        elif len_signal < self.num_samples:
            len_pad = self.num_samples - len_signal
            last_dim_padding = (0, len_pad)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _mel_to_img(self, mel):
        # Convert to numpy array
        spec = mel.squeeze()
        spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec)) * 255

        # Convert Mel Spectrogram to an image
        # TODO can this be done without numpy on cuda?
        img = Image.fromarray(spec.cpu().data.numpy())
        img = torchvision.transforms.RandomVerticalFlip(1)(img)
        img = torchvision.transforms.Resize((480, 640))(img)
        #img = torchvision.transforms.Resize(80)(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize([0.5], [0.5])(img)
        return img

    def _get_audio_sample_path(self, index):
        folder = self.annotations.iloc[index, 0]
        file = self.annotations.iloc[index, 1]
        path = os.path.join(self.audio_dir, folder, file)
        return path

    def _get_audio_sample_label(self, index):
        return str(self.annotations.iloc[index, 0]).split(" - ")[1]


if __name__ == "__main__":
    AUDIO_DIR = "D:/datasets/ESC-50"
    SAMPLE_RATE = 22050
    SECONDS = 1     # all clips should be 5 seconds in length
    NUM_SAMPLES = SAMPLE_RATE * SECONDS
    DSIZE = 1024
    N_MELS = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device {device}")
    print(f"Loading audio from {AUDIO_DIR}")
    esc = ESCDataset(AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, device, ret_type="waveform")
    print(f"There are {len(esc)} samples in the dataset")

    #import matplotlib.pyplot as plt
    #plt.pcolormesh(esc[420][0].cpu().data.squeeze())
    #plt.plot(esc[420][0].cpu().data.squeeze())
    #plt.show()
    #esc[420][0].show()