from edm import edm_sampler, EDiffusion
from networks import SongUNet, DhariwalUNet
from esc_dataset import ESCDataset
from fsc_dataset import FSCDataset
from rain_dataset import RainDataset
from rainspec_dataset import RainSpecDataset
from env_large_dataset import EnvDataset

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms.v2 import RandomCrop, PILToTensor, ToTensor, Resize, Normalize, Compose, ToImage, ToDtype
from torchvision.transforms.v2 import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

import os
import random
from datetime import datetime
from torchsummary import summary
from PIL import Image


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
    #img_dir = "D:/datasets/rain_spec"
    #audio_dir = "D:/datasets/ENV_DS-LARGE"
    #audio_dir = "D:/datasets/ENV_DS-SMOL"
    audio_dir = "D:/datasets/ENV_DS-CLEAN"
    target_sample_rate = 22050
    # 3 seconds at 22050Hz results in a 128x130 spectrogram, and we can use 128x128 resolution
    # 5 seconds at 22050Hz results in a 128x216 spectrogram, so we have to use 256x256 resolution
    # Alternatively, just random crop to 128x128 to maintain full vertical resolution!
    seconds = 3
    img_size = 128  # 256  # 32

    # Parameters
    channels = 1        # Grayscale
    batch_size = 128  # 16  # 16
    #eval_batch_size = 4

    learning_rate = 1e-4  # 1e-4
    n_steps = 100000
    sampling_steps = 32
    accumulation_steps = 1      # 16     # Option for gradient accumulation with very large data
    warmup = 5000                   # How fast we increase the learning rate for the optimizer
    resume_training = True
    resume_ckpt = "./output/_OvernightSmall/checkpoints/edm_ckpt_49999.pth"

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

    #transform = Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
    transform = RandomCrop((img_size, img_size)).to(device)
    training_data = EnvDataset(audio_dir, seconds=seconds, transform=transform)

    n_classes = len(training_data.classes)
    label_tensor = torch.tensor([i for i in range(n_classes)]).to(device)
    eval_batch_size = n_classes         # Sample 4x per class

    # DataLoader
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"DataLoader contains {len(dataloader)} images")

    # Initialize UNet and EDM Model
    unet = DhariwalUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=n_classes,
        augment_dim=9,
        model_channels=16,
        channel_mult=[1, 2, 3, 4],      # [1, 2, 3, 4]
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
        edm.model.load_state_dict(checkpoint["model_state_dict"])
        edm.ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

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
            #plt.imshow(img_batch[0].squeeze())
            #plt.show()
            img_batch = img_batch.to(device)
            label_dict = label_dict.to(device)
            loss = edm.train_one_step(img_batch, labels=label_dict) / accumulation_steps
            #loss.sum().backward()
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

        # Save model every 500 steps
        if (step % 500 == 0 or step == n_steps - 1) and step > 0:  # (step % (n_steps // 4) == 0 or step == n_steps - 1) and step > 0:
            # Save a checkpoint with parameters stored
            torch.save({
                'step': step,
                'model_state_dict': edm.model.state_dict(),
                'ema_state_dict': edm.ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, f=f"{ckpt_dir}/edm_ckpt_{step}.pth")
            # Save a model to sample from
            torch.save(edm.model.state_dict(), f=f"{ckpt_dir}/model_sample_{step}.pth")
            torch.save(edm.ema.state_dict(), f=f"{ckpt_dir}/ema_sample_{step}.pth")

        # Save sample image every 500 steps
        if (step % 500 == 0 or step == n_steps - 1) and step > 0:  # (step % (n_steps // 4) == 0 or step == n_steps - 1) and step > 0:
            edm.model.eval()  # Switch to eval mode to take a sample
            latents = torch.randn([eval_batch_size, channels, img_size, img_size], device=device)#.to(device)
            #sample = edm_sampler(edm, latents, class_labels=label_tensor, num_steps=sampling_steps).detach().cpu()
            #torchvision.utils.save_image(tensor=(sample / 2 + 0.5).clamp(0, 1),
            #                             fp=f"{sample_dir}/image_step_{step}.png")
            sample = edm_sampler(edm, latents, class_labels=label_tensor, num_steps=sampling_steps)
            # is the * 50 necessary? 2455 max luminance value
            images_np = sample.to(torch.float32).cpu().numpy()
            idx = 0
            # Split samples by class
            for img, label in zip(images_np, label_tensor):
                if not os.path.exists(f"{sample_dir}/{label}"):
                    os.makedirs(f"{sample_dir}/{label}", exist_ok=True)
                im = Image.fromarray(img.squeeze()).convert("F")
                im.save(f"{sample_dir}/{label}/image_step_{step}_{idx}.tiff")
                #im.save(f"{sample_dir}/image_step_{step}_{idx}.tiff")
                idx += 1

            edm.model.train()  # Back to training mode

        # Save fully trained models
        if step == n_steps - 1:
            torch.save(edm.model.state_dict(), f=f"{models_dir}/model_{step}_{run_id}.pth")
            torch.save(edm.ema.state_dict(), f=f"{models_dir}/ema_{step}_{run_id}.pth")

    print("done done")

