## Environmental Audio Diffusion
This project is an experiment in diffusion-based generation of outdoor soundscapes (e.g. "Birds chirping as wind blows through trees", or "A crackling campfire on the beach"). This concept was originally intended for dynamically generating ambient soundscapes for use as background audio in TTRPG's.

This repository contains code used to train, sample, and playback various sounds of the following classes:

- Birds
- Crickets
- Fire
- Fireworks
- Insects
- Ocean
- Rain
- Raindrops
- Silence
- Thunderstorm
- Wind

The audio files (.wav and .ogg) used for training came from the ESC-50, FSC22, and FSD50K datasets.

The UNet and processes used in the diffusion model stem from those discussed in "Elucidating the Design Space of Diffusion-Based Generative Models" (Tero Karras, et al.)

### Citations
Datasets:

http://dx.doi.org/10.1145/2733373.2806390

https://doi.org/10.34740/KAGGLE/DSV/4213460

https://doi.org/10.5281/zenodo.4060432

Research papers referenced:

http://doi.org/10.48550/arXiv.2206.00364

http://doi.org/10.48550/arXiv.2311.08667

http://doi.org/10.48550/arXiv.2312.02696


Code repositories used for learning and as reference:

https://github.com/nvlabs/edm

https://github.com/yuanzhi-zhu/mini_edm

