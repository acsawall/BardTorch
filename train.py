import torch
from torch import nn
import numpy as np
from sklearn import preprocessing
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader
from esc_dataset_spec import ESCDataset
from cnn import CNNNetwork

# https://www.youtube.com/watch?v=4p0G6tgNLis&ab_channel=ValerioVelardo-TheSoundofAI
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


def create_dataloader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(targets)
        targets = torch.as_tensor(targets)
        targets = targets.to(device)

        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("---------------------")
    print("Training is done!")


if __name__ == "__main__":
    # Init Dataset
    AUDIO_DIR = "D:/datasets/ESC-50"
    SAMPLE_RATE = 22050
    SECONDS = 1  # all clips should be 5 seconds in length
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

    esc = ESCDataset(AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    # Create DataLoader
    train_data_loader = create_dataloader(esc, BATCH_SIZE)

    # Build Model
    cnn = CNNNetwork().to(device)
    print(cnn)

    # Loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train Model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    # Store Model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored at cnn.pth")

