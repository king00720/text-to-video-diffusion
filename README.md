# Minimal Text-to-Video Local Demo

This project builds a tiny text-to-video diffusion model that trains on synthetic moving shapes. It is intentionally small and fast enough to run on a laptop CPU for a local demo.

What it does:
- Generates short synthetic videos of moving objects such as circles, squares, and triangles
- Conditions generation on text prompts like `red circle moving right`
- Trains a lightweight 3D diffusion model in PyTorch
- Saves a GIF after sampling

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model for a short demo run:

```bash
python train.py --epochs 10 --batch-size 8
```

3. Generate a sample video from a prompt:

```bash
python sample.py --prompt "blue square moving left"
```

This creates `outputs/demo.gif`.

## Project files

- `model.py` — synthetic data generator, diffusion utilities, and model definition
- `train.py` — training loop
- `sample.py` — text-conditioned generation and GIF export
- `requirements.txt` — dependencies

## Model design

This is a toy but real diffusion model:
- text is converted into token embeddings
- a time embedding is added
- the noisy video tensor is processed through a 3D convolutional network
- text and time conditioning are injected using FiLM-style modulation
- the model predicts added noise

This is much smaller than production text-to-video systems, but it is a valid local demo and a good starting point for scaling.

## Notes

- The dataset is synthetic rather than real-world video data.
- This is meant for learning and demos, not a production-grade generative video model.
- You can expand the vocabulary and prompts to make it more expressive.
