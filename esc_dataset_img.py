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


class ESCDataset(Dataset):

    def __init__(self, audio_dir, transform, target_sample_rate, num_samples, device):
        self.audio_dir = audio_dir
        annotations = pd.DataFrame(columns=["folder", "file"])
        for folder in os.listdir(audio_dir):
            if os.path.isdir(audio_dir + "/" + folder):
                for file in os.listdir(audio_dir + "/" + folder):
                    annotations = pd.concat([annotations, pd.DataFrame({"folder": [folder], "file": [file]})], ignore_index=True)
        self.annotations = annotations
        self.device = device
        self.transform = transform.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        #print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Load audio from disk
        signal, sr = torchaudio.load(audio_sample_path)
        # Apply transforms
        signal = self._resample(signal, sr)
        signal = self._mix_to_mono(signal)
        signal = self._resize_signal_tensor(signal)
        signal = signal.to(self.device)
        mel_spec = self.transform(signal)

        # Scale logarithmically
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        img = self._mel_to_numpy(mel_spec)

        return img, label

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            # Create resampler method
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

    def _mel_to_numpy(self, mel):
        # Convert to numpy array
        spec = mel.squeeze()
        spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec)) * 255

        # Convert Mel Spectrogram to an image
        # TODO can this be done without numpy on cuda?
        img = Image.fromarray(spec.cpu().data.numpy())
        img = torchvision.transforms.RandomVerticalFlip(1)(img)
        img = torchvision.transforms.Resize((480, 640))(img)
        return img

    def _get_audio_sample_path(self, index):
        folder = self.annotations.iloc[index, 0]
        file = self.annotations.iloc[index, 1]
        path = os.path.join(self.audio_dir, folder, file)
        return path

    def _get_audio_sample_label(self, index):
        #return torch.as_tensor(list(str(self.annotations.iloc[index, 0]).split(" - ")[1]))
        #return self.annotations.iloc[index, 0]
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

    # Setup mel_spectrogram to be used as a transform
    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=DSIZE,
        hop_length=DSIZE // 4,
        n_mels=N_MELS
    )

    print(f"Loading audio from {AUDIO_DIR}")
    esc = ESCDataset(AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(esc)} samples in the dataset")

    #mel_spec, label = esc[420]
    #ImageShow.show(mel_spec)
    #print(label)