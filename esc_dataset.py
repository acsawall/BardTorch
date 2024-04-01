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
from PIL import Image


AUDIO_DIR = "D:/datasets/ESC-50"


class ESCDataset(Dataset):

    def __init__(self, audio_dir, transform, target_sample_rate):
        self.audio_dir = audio_dir
        annotations = pd.DataFrame(columns=["folder", "file"])
        for folder in os.listdir(audio_dir):
            if os.path.isdir(audio_dir + "/" + folder):
                for file in os.listdir(audio_dir + "/" + folder):
                    annotations = pd.concat([annotations, pd.DataFrame({"folder": [folder], "file": [file]})], ignore_index=True)
        self.annotations = annotations
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Load audio from disk
        signal, sr = torchaudio.load(audio_sample_path)
        # Apply transforms
        resampled_signal = self._resample(signal, sr)
        mono_signal = self._mix_to_mono(resampled_signal)
        mel_spec = self.transform(mono_signal)

        # Scale logarithmically
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        img = self._mel_to_numpy(mel_spec)

        #return signal, resampled_signal, mono_signal, spec, img, label
        #return signal, img, label
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

    def _mel_to_numpy(self, mel):
        # Convert to numpy array
        spec = mel.squeeze().numpy()
        spec = (spec - spec.min()) / (spec.max() - spec.min()) * 255
        spec = spec.astype('uint8')

        # Convert Mel Spectrogram to an image
        img = Image.fromarray(spec)
        img = torchvision.transforms.RandomVerticalFlip(1)(img)
        img = torchvision.transforms.Resize(((480, 640)))(img)
        return img

    def _get_audio_sample_path(self, index):
        folder = self.annotations.iloc[index, 0]
        file = self.annotations.iloc[index, 1]
        path = os.path.join(self.audio_dir, folder, file)
        return path

    def _get_audio_sample_label(self, index):
        # TODO change label when testing sample pulls
        return str(self.annotations.iloc[index, 0]).split(" - ")[1]
        #return str(self.annotations.iloc[index])


if __name__ == "__main__":
    SAMPLE_RATE = 16000
    DSIZE = 1024

    # Setup mel_spectrogram to be used as a transform
    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=DSIZE,
        hop_length=DSIZE // 2,
        n_mels=64
    )

    print(f"Loading audio from {AUDIO_DIR}")
    esc = ESCDataset(AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    print(f"There are {len(esc)} samples in the dataset")

    signal, img, label = esc[420]
    import matplotlib.pyplot as plt
    #plt.plot(signal.squeeze())
    #plt.show()

    #plt.specgram(signal.squeeze())
    #plt.title("Original signal")
    #plt.show()

    '''plt.specgram(resampled_signal.squeeze())
    plt.title("Resampled signal")
    plt.show()

    plt.specgram(mono_signal.squeeze())
    plt.title("Mono signal")
    plt.show()

    plt.specgram(spec)
    plt.title("Normalized signal")
    plt.show()'''

    plt.imshow(img)
    plt.title(label)
    plt.show()

    # Test pull sample from dataset
    import sounddevice as sd
    import soundfile as sf

    #sample_no = 789
    #signal, label = esc[sample_no]
    #print(f"\nLoaded test sample {sample_no}:")
    #print(label)
    #import matplotlib.pyplot as plt
    #plt.specgram(signal.squeeze())
    #plt.show()
    sf.write("testing/test.wav", signal.squeeze(), 44100)
    #sd.play(signal, 44100)
    #sd.wait()
