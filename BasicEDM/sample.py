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
from networks import SongUNet, DhariwalUNet


if __name__ == "__main__":
    model_path = "./output/train_mnist_2024-04-19_204920/checkpoints/ema_sample_250.pth"
    #labels = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    #labels = ['4 - four', '5 - five', '6 - six', '7 - seven',
    #          '8 - eight', '9 - nine']
    #labels = ['0 - zero', '1 - one', '2 - two']
    img_size = 256
    channels = 1
    classes = 0
    eval_batch_size = 8
    sampling_steps = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    labels = torch.tensor([6, 9, 6, 9, 4, 2, 0, 1]).to(device)

    # Set random seed
    seed = random.randint(1, 9999)  # 69
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    '''unet = SongUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=classes,
        augment_dim=9,
        model_channels=16,
        channel_mult=[1, 2, 3, 4],
        num_blocks=1,
        attn_resolutions=[0]
    )'''
    unet = DhariwalUNet(
        img_resolution=img_size,
        in_channels=channels,
        out_channels=channels,
        label_dim=0,
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
    l = torch.load(model_path, map_location=device)
    edm.ema.load_state_dict(l)
    for p in edm.ema.parameters():
        p.requires_grad = False
    edm.ema.eval()
    latents = torch.randn([eval_batch_size, channels, img_size, img_size]).to(device).float()
    sample = edm_sampler(edm, latents, use_ema=True, class_labels=labels, num_steps=sampling_steps).detach().cpu()

    sample_dir = f"output/samples"
    os.makedirs(sample_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torchvision.utils.save_image(tensor=(sample / 2 + 0.5).clamp(0, 1), fp=f"{sample_dir}/sample_{timestamp}.png")



