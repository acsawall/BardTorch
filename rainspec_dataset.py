import librosa
import librosa.feature
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
from tqdm import tqdm
from audio_to_spec import get_melspec_db, spec_to_img, load_audio_from_tiff
from skimage import io, transform
import matplotlib.pyplot as plt

class RainSpecDataset(Dataset):
    def __init__(
            self,
            image_dir,
            transform = None
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.files = []
        for file in tqdm(os.listdir(image_dir)):
            self.files.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.files[index])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    dir = "D:/datasets/rain_spec"
    rsd = RainSpecDataset(dir)

    m = rsd[0]
    img = np.array(m)
    img *= 50
    aud = librosa.feature.inverse.mel_to_audio(img, sr=22050, n_fft=2048, hop_length=512)
    plt.plot(aud)
    plt.show()
    import sounddevice as sd
    sd.play(aud, 22050)
    sd.wait()
