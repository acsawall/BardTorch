from edm import edm_sampler, EDiffusion

import os
import random
from datetime import datetime
import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from networks import DhariwalUNet
from PIL import Image

if __name__ == "__main__":
    model_path = "../model/ema_sample_299999.pth"

    img_res = 128
    img_width = 256
    img_height = 128
    channels = 1
    classes = 11
    eval_batch_size = 8
    sampling_steps = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    label_tensor = torch.tensor([0 for _ in range(eval_batch_size)]).to(device)

    # Set random seed
    seed = random.randint(1, 9999)  # 69
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    unet = DhariwalUNet(
        img_resolution=img_res,
        in_channels=channels,
        out_channels=channels,
        label_dim=classes,
        augment_dim=9,
        model_channels=16,
        channel_mult=[1, 2, 3, 4],
        num_blocks=1,
        attn_resolutions=[0],
        label_dropout=0.0
    )

    edm = EDiffusion(
        model=unet,
        device=device
    )
    l = torch.load(model_path, map_location=device)
    edm.ema.load_state_dict(l)
    for p in edm.ema.parameters():
        p.requires_grad = False
    edm.ema.eval()
    latents = torch.randn([eval_batch_size, channels, img_height, img_width]).to(device).float()
    # sample = edm_sampler(edm, latents, use_ema=True, class_labels=labels, num_steps=sampling_steps).detach().cpu()
    sample = edm_sampler(edm, latents, class_labels=label_tensor, num_steps=sampling_steps)
    images_np = sample.to(torch.float32).cpu().numpy()
    sample_dir = f"output/samples"
    os.makedirs(sample_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = 0
    # Split samples by class
    for img, label in zip(images_np, label_tensor):
        im = Image.fromarray(img.squeeze()).convert("F")
        im.save(f"{sample_dir}/image_{label}_{idx}.tiff")
        # im.save(f"{sample_dir}/image_step_{step}_{idx}.tiff")
        idx += 1
    # torchvision.utils.save_image(tensor=(sample / 2 + 0.5).clamp(0, 1), fp=f"{sample_dir}/sample_{timestamp}.png")
