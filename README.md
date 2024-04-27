### Environmental Audio Diffusion
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

