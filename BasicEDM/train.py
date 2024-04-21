from edm import edm_sampler, EDiffusion
from networks import SongUNet, DhariwalUNet
from esc_dataset import ESCDataset
from fsc_dataset import FSCDataset
from rain_dataset import RainDataset
from rainspec_dataset import RainSpecDataset

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms.v2 import PILToTensor, ToTensor, Resize, Normalize, Compose, ToImage, ToDtype
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import os
import random
from datetime import datetime
from torchsummary import summary


def download_mnist_training(img_size):
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                Resize(img_size),
                Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                Normalize((0.5,), (0.5,))
            ]
        )
    )
    return train_data


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Dataset Params
    #audio_dir = "D:/datasets/ESC-50"
    #audio_dir = "D:/datasets/rain"
    #audio_dir = "D:/datasets/ESC-mini"
    #audio_dir = "D:/datasets/FSC22/Audio Wise V1.0"
    #metadata_path = "D:/datasets/FSC22/Metadata V1.0 FSC22.csv"
    img_dir = "D:/datasets/rain_spec"
    target_sample_rate = 22050
    seconds = 5
    ret_type = "image"

    # Parameters
    channels = 1        # Grayscale
    batch_size = 8  # 16
    eval_batch_size = 16
    img_size = 256  # 32
    learning_rate = 1e-4  # 1e-4
    n_steps = 5000
    sampling_steps = 18
    accumulation_steps = 1      # 16     # Option for gradient accumulation with very large datasets
    warmup = 1000                   # How fast we increase the learning rate for the optimizer
    resume_training = False
    resume_ckpt = "./output/train_mnist_2024-04-19_163852/checkpoints/ema_ckpt_4999.pth"

    # Setup output directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = f"output/train_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    sample_dir = f"{outdir}/samples"
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = f"{outdir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    models_dir = f"output/complete_models"
    os.makedirs(models_dir, exist_ok=True)

    # Set random seed
    # TODO make this set-able
    seed = 69
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    #training_data = download_mnist_training(img_size)
    #mnist = download_mnist_training(img_size)
    #training_data = ESCDataset(audio_dir=audio_dir, target_sample_rate=target_sample_rate, num_samples=target_sample_rate * seconds, image_resolution=img_size, device=device, ret_type=ret_type)
    #training_data = RainDataset(audio_dir, target_sample_rate, target_sample_rate * seconds, img_size, device)
    training_data = RainSpecDataset(img_dir,
                                    torchvision.transforms.Compose(
                                        [
                                            Resize(img_size),
                                            Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                                            Normalize((0.5,), (0.5,))
                                        ]
                                    ))
    #training_data = FSCDataset(audio_dir=audio_dir, metadata_path=metadata_path, target_sample_rate=target_sample_rate, num_samples=target_sample_rate*seconds, image_resolution=img_size, device=device)

    #n_classes = len(training_data.classes)
    # Labels to use for mid-training sampling
    #label_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]).to(device)
    label_tensor = None #torch.tensor([1, 1, 1, 1, 1, 2, 2, 4, 4, 10, 10, 10, 11, 11, 14, 14]).to(device)

    # DataLoader
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"DataLoader contains {len(dataloader)} images")

    # Initialize UNet and EDM Model
    '''unet = SongUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=len(classes),
        augment_dim=9,
        model_channels=64,  #16,
        channel_mult=[1, 2, 3, 4],  #[1, 2, 3, 4],
        resample_filter=[1, 1],
        num_blocks=4,
        attn_resolutions=[0]
    )'''
    unet = DhariwalUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=0, # n_classes,
        augment_dim=9,
        model_channels=64,
        channel_mult=[1, 2, 3, 4],
        num_blocks=1,
        attn_resolutions=[0]
    )
    edm = EDiffusion(
        model=unet,
        device=device
    )

    optimizer = Adam(edm.model.parameters(), lr=learning_rate)
    train_loss = 0
    start_step = 0
    if resume_training:
        checkpoint = torch.load(resume_ckpt, map_location=device)
        edm.ema.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss = checkpoint["train_loss"]
        start_step = checkpoint["step"]

    edm.model.train()
    # Training Loop
    print("### Starting Training ###")
    for step in range(start_step, n_steps):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        # TODO Gradient accumulation may not be needed for the ESC dataset
        for _ in range(accumulation_steps):
            try:
                img_batch, label_dict = next(data_iterator)
            except Exception:
                data_iterator = iter(dataloader)
                img_batch, label_dict = next(data_iterator)
            #import matplotlib.pyplot as plt
            #plt.imshow(img_batch[0].squeeze())
            #plt.show()
            img_batch = img_batch.to(device)
            label_dict = label_dict.to(device)
            loss = edm.train_one_step(img_batch, labels=label_dict) / accumulation_steps
            loss.backward()
            batch_loss += loss

        # Update optimizer and unet weights
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * min(step / warmup, 1)
        for param in unet.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        train_loss += batch_loss.detach().item()

        edm.update_exp_moving_avg(step=step, batch_size=batch_size)

        # Print progress every 250 steps
        if step % 250 == 0 or step == n_steps - 1:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} -> Step: {step:08d}; Current Learning Rate: {optimizer.param_groups[0]['lr']:0.6f}; Average Loss: {train_loss / (step + 1):0.10f}; Batch Loss: {batch_loss.detach().item():0.10f}")

        # Save sample image every 25% complete
        if (step % (n_steps // 4) == 0 or step == n_steps - 1) and step > 0:
            edm.model.eval()    # Switch to eval mode to take a sample
            latents = torch.randn([eval_batch_size, channels, img_size, img_size]).to(device)
            sample = edm_sampler(edm, latents, class_labels=label_tensor, num_steps=sampling_steps).detach().cpu()
            torchvision.utils.save_image(tensor=(sample / 2 + 0.5).clamp(0, 1), fp=f"{sample_dir}/image_step_{step}.png")
            edm.model.train()   # Back to training mode

        # Save model checkpoints at 25%, 50%, 75%, and 100% trained
        if (step % (n_steps // 4) == 0 or step == n_steps - 1) and step > 0:
            # Save a checkpoint with parameters stored
            torch.save({
                'step': step,
                'model_state_dict': edm.ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, f=f"{ckpt_dir}/ema_ckpt_{step}.pth")
            # Save a model to load from
            torch.save(edm.ema.state_dict(), f=f"{ckpt_dir}/ema_sample_{step}.pth")
            # Test alternate save method to load/sample from
            #torch.save(dict(net=edm, optimizer_state=optimizer.state_dict()), f=f"{ckpt_dir}/ema_alternate_{step}.pth")

        # Save fully trained models
        if step == n_steps - 1:
            #torch.save(edm.model.state_dict(), f=f"{models_dir}/model_{step}_{run_id}.pth")
            torch.save(edm.ema.state_dict(), f=f"{models_dir}/ema_{step}_{run_id}.pth")

    print("done done")

