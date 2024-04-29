import torch
import torchvision
from torchvision.transforms.v2 import ToImage, Resize
from PIL import Image
import sounddevice as sd
import numpy as np
import librosa
import librosa.feature
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import noisereduce as nr            # https://github.com/timsainb/noisereduce
import soundfile as sf


"""Contains utilities for saving/loading audio from .tiff image files, as well as denoising .tiff images"""
# get_melspec_db from https://medium.com/@hasithsura/audio-classification-d37a82d6715
def get_melspec_db(fpath, sr=None, seconds=5, n_fft=2048, hop_len=512, n_mels=128, fmin=20, fmax=8300, top_db=120):
    signal, sr = librosa.load(fpath, sr=sr)
    if signal.shape[0] < seconds * sr:
        signal = np.pad(signal, int(np.ceil((seconds * sr - signal.shape[0]) / 2)))
    else:
        signal = signal[:seconds * sr]
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
    spec = librosa.power_to_db(spec, top_db=top_db)
    return spec


def save_audio_as_spec_tiff(audio_path, img_path, seconds, target_sr):
    signal, sr = librosa.load(audio_path, sr=target_sr)
    if signal.shape[0] < seconds * sr:
        signal = np.pad(signal, int(np.ceil((seconds * sr - signal.shape[0]) / 2)))
    else:
        signal = signal[:seconds * sr]
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    # Normalize, inverse mel must be *= 50
    # This is necessary in staying under the max luminance value for .tiff files
    spec /= 50
    im = Image.fromarray(spec).convert("F")
    im.save(img_path)


def load_audio_from_tiff(img_path, sr, medfilter=True):
    m = Image.open(img_path)
    img = np.array(m, dtype=np.float32)
    img *= 50
    if medfilter:
        img = sig.medfilt2d(np.abs(img), 3)
    aud = librosa.feature.inverse.mel_to_audio(img, sr=sr, n_fft=2048, hop_length=512)
    return aud


def scipy_FIR(data, sr, cutoff, filter_order, filter_type=True):
    scipy_fir = sig.firwin(filter_order + 1, cutoff, fs=sr, pass_zero=filter_type)
    scipy_filtered = sig.convolve(data, scipy_fir, mode='same')
    return scipy_filtered


def denoise(data, sr, noise_sample=None, stationary=False):
    plt.specgram(data)
    plt.show()
    #data = scipy_FIR(data, sr, 10000, 5)   # lowpass high artifacts
    reduced_noise = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample, stationary=stationary, prop_decrease=1, n_fft=2048)
    plt.specgram(reduced_noise)
    plt.show()
    return reduced_noise


'''
Denoise NonStationary: Crickets, Fireworks, Raindrops

Denoise Stationary: Ocean, Rain, Thunderstorm, Wind, Birds, Insects

Maybe don't Denoise: Rain, Wind
'''
if __name__ == "__main__":
    device = torch.device("cuda")
    src = "D:/datasets/ENV_DS-LARGE"
    dest = "D:/datasets/ENV_DS-LARGESPEC"
    target_sr = 22050
    aud_path = "D:/datasets/ENV_DS-LARGE/Thunderstorm/27074.wav"
    signal, sr = librosa.load(aud_path, sr=target_sr)
    seconds = 5

    #silent_noise = "BasicEDM/output/_OvernightLargeSpec/samples/8/image_step_95500_8.tiff"
    noisy_noise = "noisefilters/noisy.tiff"
    quiet_noise = "noisefilters/quiet.tiff"
    medium_noise = "noisefilters/medium.tiff"
    noise = load_audio_from_tiff(quiet_noise, target_sr, True)
    #sd.play(noise, target_sr)
    #sd.wait()
    '''img_path = os.path.join("C:/Users/aidan/Desktop/Presentation Samples/Crickets/image_1_0.tiff")
    aud = load_audio_from_tiff(img_path, target_sr, True)
    aud = denoise(aud, target_sr, noise, True)
    plt.plot(aud)
    plt.show()
    #sd.play(aud, target_sr)
    #sd.wait()
    sf.write(f"cricket_chirp.wav", aud, target_sr)'''

    for file in os.listdir("C:/Users/aidan/Desktop/Presentation Samples/Insects"):
        img_path = os.path.join("C:/Users/aidan/Desktop/Presentation Samples/Insects", file)
        aud = load_audio_from_tiff(img_path, target_sr, True)
        aud = denoise(aud, target_sr, noise, True)
        plt.plot(aud)
        plt.show()
        #sd.play(aud, target_sr)
        #sd.wait()
        sf.write(f"insect_{file}.wav", aud, target_sr)

