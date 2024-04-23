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
from scipy import signal
import matplotlib.pyplot as plt
import noisereduce as nr            # https://github.com/timsainb/noisereduce


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
    plt.pcolormesh(spec)
    plt.show()
    im = Image.fromarray(spec).convert("F")
    im.save(img_path)


def load_audio_from_tiff(img_path, sr):
    m = Image.open(img_path)
    img = np.array(m)
    #img = np.flip(img, axis=0)
    img *= 50
    aud = librosa.feature.inverse.mel_to_audio(img, sr=sr, n_fft=2048, hop_length=512)
    return aud


def butter_lowpass_filter(data, cutoff, sr, order=5):
    b, a = signal.butter(order, cutoff, fs=sr, btype="lowpass", analog=False, output='ba')
    y = signal.lfilter(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, sr, order=5):
    b, a = signal.butter(order, cutoff, fs=sr, btype="highpass", analog=False, output='ba')
    y = signal.lfilter(b, a, data)
    return y


def scipy_FIR(data, sr, cutoff, filter_order, filter_type=True):
    scipy_fir = signal.firwin(filter_order + 1, cutoff, fs=sr, pass_zero=filter_type)
    scipy_filtered = signal.convolve(data, scipy_fir, mode='same')
    return scipy_filtered


def denoise(data, sr, noise_sample):
    #f, t, spec = signal.stft(data, sr, nfft=2048)
    #plt.pcolormesh(t, f, np.abs(spec))
    #plt.show()
    plt.specgram(data)
    plt.show()
    reduced_noise = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample, stationary=True, prop_decrease=1, n_fft=2048)
    reduced_noise = nr.reduce_noise(y=reduced_noise, sr=sr, y_noise=noise_sample, stationary=False, prop_decrease=1, n_fft=2048)
    return reduced_noise


if __name__ == "__main__":
    device = torch.device("cuda")
    src = "D:/datasets/ENV_DS-SMOL"
    dest = "D:/datasets/ENV_DS-SMOLSPEC"
    target_sr = 22050

    '''for file in os.listdir(dest):
        img_path = os.path.join(dest, file)
        aud = load_audio_from_tiff(img_path, target_sr)
        #plt.plot(aud)
        #plt.show()
        sd.play(aud, target_sr)
        sd.wait()'''

    # This worked ok for insect denoising
    #noise_path = os.path.join("BasicEDM/output/_FirstSmolRun/samples/0/image_step_1500_0.tiff")
    # Fireworks Noise
    noise_path = os.path.join("BasicEDM/output/_OvernightClean/samples/0/image_step_2500_0.tiff")
    #img_path = "test.tiff"
    noise = load_audio_from_tiff(noise_path, target_sr)
    #sd.play(noise, target_sr)
    #sd.wait()
    for file in os.listdir("BasicEDM/output/samples"):
    #for file in os.listdir(os.path.join("D:/datasets/ENV_DS-CLEANSPEC/Fireworks")):
        img_path = os.path.join("BasicEDM/output/samples", file)
        #img_path = os.path.join("D:/datasets/ENV_DS-CLEANSPEC/Fireworks", file)
        aud = load_audio_from_tiff(img_path, target_sr)
        aud = denoise(aud, target_sr, noise) * 10
        plt.specgram(aud)
        plt.show()
        sd.play(aud, target_sr)
        sd.wait()

    #save_audio_as_spec_tiff("D:/datasets/ENV_DS-CLEAN/Fireworks/1-25777-A.ogg", "test.tiff", 3, target_sr)
    #import soundfile as sf
    #sf.write("test_rain.wav", aud, target_sr)

    #idx = 0
    #for folder in os.listdir(src):
    #    for file in os.listdir(os.path.join(src, folder)):
    #        save_audio_as_spec_tiff(os.path.join(src, folder, file), os.path.join(dest, folder, f"{idx}_{target_sr}.tiff"), 3, target_sr)
    #        idx += 1

