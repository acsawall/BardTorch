import librosa
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


class EnvDataset(Dataset):

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(self.classes)}

    '''Assumes a file structure of:
        root_dir
            class1
                audio files
            class2
                audio files
            ...
    '''
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
        for folder in tqdm(os.listdir(self.root_dir), desc="Loading Dataset"):
            self.classes.append(folder)
            if os.path.isdir(root_dir + "/" + folder):
                for file in os.listdir(root_dir + "/" + folder):
                    fp = os.path.join(root_dir, folder, file)
                    signal, sr = librosa.load(fp, sr=self.target_sr)
                    signal = self._resize_signal(signal)
                    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
                    # Uniformly "normalize" in case images are converted to .tiff, to stay under the max luminance value
                    spec /= 50
                    # Convert to float PIL image
                    pil = Image.fromarray(spec).convert("F")
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

    # Extend audio shorter than self.seconds, mainly applies here to files from FSD50K that are under 3-5 seconds
    def _resize_signal(self, signal):
        if signal.shape[0] < self.seconds * self.target_sr:
            signal = np.pad(signal, int(np.ceil((self.seconds * self.target_sr - signal.shape[0]) / 2)))
        return signal


if __name__ == "__main__":
    root_dir = "D:/datasets/ENV_DS-LARGE"
    print("\n\n")
    transform = RandomCrop((128, 128)).to(device=torch.device("cuda"))
    env = EnvDataset(root_dir=root_dir, seconds=3, transform=transform)
    print(f"Dataset at '{root_dir}' contains {len(env)} images across {len(env.classes)} classes\n\n")

    for i in range(5):
        m = env[i][0]
        l = env[i][1]
        img = np.array(m.cpu().data.squeeze()) * 50
        plt.imshow(img)
        plt.show()
        aud = librosa.feature.inverse.mel_to_audio(img, sr=22050, n_fft=2048, hop_length=512)

        plt.plot(aud)
        plt.show()
        import sounddevice as sd
        sd.play(aud, 22050)
        sd.wait()
        print(aud.size)
