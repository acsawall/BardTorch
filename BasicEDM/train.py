# Diffusion model and networks
from edm import edm_sampler, EDiffusion
from networks import DhariwalUNet

# Datasets
from env_large_dataset import EnvDataset
from env_spec_dataset import EnvSpecDataset

# PyTorch
import torch
from torchvision.transforms.v2 import RandomCrop
from torch.utils.data import DataLoader
from torch.optim import Adam

# Misc.
import os
import random
from datetime import datetime
from PIL import Image

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Dataset Params
    image_dir = "D:/datasets/ENV_DS-LARGESPEC"      # ENV_DS-LARGE already in .tiff format
    target_sample_rate = 22050
    img_size = 128

    # Parameters
    channels = 1                    # Grayscale
    batch_size = 128

    learning_rate = 1e-4
    n_steps = 300000
    sampling_steps = 32
    warmup = 5000                   # How fast we increase the learning rate for the optimizer

    # Options to resume training from a .pth or .pt checkpoint
    resume_training = False
    resume_ckpt = "./output/_OvernightLargeSpec240k/checkpoints/edm_ckpt_243000.pth"

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

    # Set seed
    seed = 69
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    transform = RandomCrop((img_size, img_size)).to(device)
    #training_data = EnvDataset(audio_dir, seconds=seconds, transform=transform)
    training_data = EnvSpecDataset(root_dir=image_dir, target_sr=target_sample_rate, transform=transform, device=device)
    n_classes = len(training_data.classes)

    # Sample one of each class while training
    label_tensor = torch.tensor([i for i in range(n_classes)]).to(device)
    eval_batch_size = n_classes

    # DataLoader
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"DataLoader contains {len(dataloader)} images")

    # Initialize UNet and EDM Model
    unet = DhariwalUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=n_classes,
        augment_dim=9,                  # Doesn't matter without an augment pipeline
        model_channels=16,
        channel_mult=[1, 2, 3, 4],      # [1, 2, 3, 4] or [1, 2, 2, 2]
        num_blocks=1,                   # Number residual blocks
        attn_resolutions=[0]
    )
    edm = EDiffusion(
        model=unet,
        device=device
    )

    optimizer = Adam(edm.model.parameters(), lr=learning_rate)
    train_loss = 0
    start_step = 0

    # Load model from checkpoint
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
        # Process image batch
        try:
            img_batch, label_dict = next(data_iterator)
        # Throws NameError and StopIteration
        except Exception:
            data_iterator = iter(dataloader)
            img_batch, label_dict = next(data_iterator)

        img_batch = img_batch.to(device)
        label_dict = label_dict.to(device)
        loss = edm.train_one_step(img_batch, labels=label_dict)
        loss.backward()

        # Update optimizer and unet weights
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * min(step / warmup, 1)
        for param in unet.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        train_loss += loss

        edm.update_exp_moving_avg(step=step, batch_size=batch_size)

        # Print progress every 250 steps
        if step % 250 == 0 or step == n_steps - 1:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} -> Step: {step:08d}; Current Learning Rate: {optimizer.param_groups[0]['lr']:0.6f}; Average Loss: {train_loss / (step + 1):0.10f}; Batch Loss: {loss:0.10f}")

        # Save model every 500 steps
        if (step % 500 == 0 or step == n_steps - 1) and step > 0:
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
        if (step % 500 == 0 or step == n_steps - 1) and step > 0:
            edm.model.eval()  # Switch to eval mode to take a sample
            # Init pure noise samples
            latents = torch.randn([eval_batch_size, channels, img_size, img_size], device=device)
            sample = edm_sampler(edm, latents, class_labels=label_tensor, num_steps=sampling_steps)
            images_np = sample.to(torch.float32).cpu().numpy()
            idx = 0
            # Split samples by class
            for img, label in zip(images_np, label_tensor):
                if not os.path.exists(f"{sample_dir}/{label}"):
                    os.makedirs(f"{sample_dir}/{label}", exist_ok=True)
                im = Image.fromarray(img.squeeze()).convert("F")
                im.save(f"{sample_dir}/{label}/image_step_{step}_{idx}.tiff")
                idx += 1

            edm.model.train()  # Back to training mode

        # Save fully trained models
        if step == n_steps - 1:
            torch.save(edm.model.state_dict(), f=f"{models_dir}/model_{step}_{run_id}.pth")
            torch.save(edm.ema.state_dict(), f=f"{models_dir}/ema_{step}_{run_id}.pth")

    print("done done")

