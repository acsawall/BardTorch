import librosa
import librosa.feature
import skimage.io
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
import torchaudio.io
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms.v2 import PILToTensor, ToTensor, Resize, Normalize, Compose, ToImage, ToDtype

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
        self.data = []
        for file in tqdm(os.listdir(image_dir)):
            img_path = os.path.join(self.image_dir, file)
            image = Image.open(img_path)
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        #img_tensor = torchvision.transforms.ToTensor()(image)
        #img_tensor *= 50        # TODO should this be done for training or sampling?
        #plt.pcolormesh(image.cpu().data.squeeze())
        #plt.show()
        return image, 0


if __name__ == "__main__":
    dir = "D:/datasets/rain_spec"
    rsd = RainSpecDataset(dir, torchvision.transforms.Compose(
                                        [
                                            Resize(256),
                                            Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                                            #Normalize((0.5,), (0.5,))
                                        ]
                                    ))

    m = rsd[0][0]
    img = np.array(m.squeeze())
    img *= 50
    aud = librosa.feature.inverse.mel_to_audio(img, sr=22050, n_fft=2048, hop_length=512)
    plt.plot(aud)
    plt.show()
    import sounddevice as sd
    sd.play(aud, 22050)
    sd.wait()
    print(aud.size)
