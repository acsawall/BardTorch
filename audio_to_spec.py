import torch
import torchaudio
import torchvision
from torchaudio.transforms import Resample, MelSpectrogram, Spectrogram
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from PIL import Image
import sounddevice as sd
import skimage.io
import numpy as np
import librosa
import librosa.feature
import os
import matplotlib.pyplot as plt


# Helper methods from https://medium.com/@hasithsura/audio-classification-d37a82d6715
def get_melspec_db(fpath, sr=None, seconds=5, n_fft=2048, hop_len=512, n_mels=128, fmin=20, fmax=8300, top_db=120):
    signal, sr = librosa.load(fpath, sr=sr)
    if signal.shape[0] < seconds * sr:
        signal = np.pad(signal, int(np.ceil((seconds * sr - signal.shape[0]) / 2)))
    else:
        signal = signal[:seconds * sr]
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    spec = librosa.power_to_db(spec, top_db=top_db)
    return spec


def spec_to_img(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def save_audio_as_spec_tiff(audio_path, img_path, seconds, target_sr):
    signal, sr = librosa.load(audio_path, sr=target_sr)
    if signal.shape[0] < seconds * sr:
        signal = np.pad(signal, int(np.ceil((seconds * sr - signal.shape[0]) / 2)))
    else:
        signal = signal[:seconds * sr]
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    spec /= 50      # Normalize, inverse mel must be *= 50
    im = Image.fromarray(spec).convert("F")
    im.save(img_path)


def load_audio_from_tiff(img_path, sr):
    m = Image.open(img_path)
    img = np.array(m)
    img *= 50
    aud = librosa.feature.inverse.mel_to_audio(img, sr=sr, n_fft=2048, hop_length=512)
    return aud


if __name__ == "__main__":
    device = torch.device("cuda")
    src = "D:/datasets/rain"
    dest = "D:/datasets/rain_spec"
    target_sr = 22050

    '''for file in os.listdir(dest):
        img_path = os.path.join(dest, file)
        aud = load_audio_from_tiff(img_path, target_sr)
        #plt.plot(aud)
        #plt.show()
        sd.play(aud, target_sr)
        sd.wait()'''

    img_path = os.path.join("BasicEDM/output/train_2024-04-21_150854/samples", "image_step_9999_0.tiff")
    aud = load_audio_from_tiff(img_path, target_sr)
    sd.play(aud, target_sr)
    sd.wait()
    import soundfile as sf
    sf.write("test_rain.wav", aud, target_sr)

    #idx = 0
    #for file in os.listdir(src):
    #    save_audio_as_spec_tiff(os.path.join(src, file), os.path.join(dest, f"{idx}_{target_sr}.tiff"), 5, target_sr)
    #    idx += 1

