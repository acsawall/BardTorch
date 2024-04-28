import librosa.feature
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.v2 import PILToTensor, Resize, ToImage, RandomCrop

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# This class functions the same as env_large_dataset.py, but reads in .tiff spectrograms instead of audio files
class EnvSpecDataset(Dataset):

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(self.classes)}

    def __init__(
            self,
            root_dir,
            target_sr=22050,
            seconds=1,            # Length of spectrograms to pad/trim to
            transform=None,
            device=torch.device("cuda")
    ):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.seconds = seconds
        self.transform = transform
        self.classes = []
        self.data = []
        self.labels = []
        self.device = device
        for folder in tqdm(os.listdir(self.root_dir), desc="Loading Data Classes..."):
            self.classes.append(folder)
            if os.path.isdir(root_dir + "/" + folder):
                for file in os.listdir(root_dir + "/" + folder):
                    fp = os.path.join(root_dir, folder, file)
                    # Convert to PIL image
                    pil = Image.open(fp)
                    # Convert to torch image tensor
                    image = ToImage()(pil)
                    # Shrink vertical access to prevent RandomCrop from distorting frequencies
                    image = Resize(128, interpolation=torchvision.transforms.v2.InterpolationMode.BICUBIC,
                                   antialias=True)(image)
                    self.data.append(image)
                    self.labels.append(folder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[self.labels[index]]


if __name__ == "__main__":
    root_dir = "D:/datasets/ENV_DS-LARGESPEC"
    #transform = Resize((128, 128), interpolation=torchvision.transforms.v2.InterpolationMode.BICUBIC, antialias=True)      # W=216 px
    transform = RandomCrop((128, 128)).to(device=torch.device("cuda"))
    env = EnvSpecDataset(root_dir=root_dir, seconds=3, transform=transform)
    for i in range(5):
        m = env[i][0]
        l = env[i][1]
        img = np.array(m.cpu().data.squeeze())
        plt.imshow(img)
        plt.show()
        aud = librosa.feature.inverse.mel_to_audio(img, sr=22050, n_fft=2048, hop_length=512)
        plt.plot(aud)
        plt.show()
        import sounddevice as sd
        sd.play(aud, 22050)
        sd.wait()
        print(aud.size)
