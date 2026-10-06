# Paper2 Reproduction

This repository contains the code and workflows for the paper:
"The Calibration of Words: A Comparative Study of Three Prompt Weight Syntaxes..."

## Environment

pip install -r requirements.txt

## Data

- `data/prompts.json`: 60 main prompts
- `data/two_word_prompts.json`: 15 two-word prompts
- `data/calibration_prompts.json`: calibration references

## Reproduction steps

1. Generate images:
   python src/generate_diffusers.py
   python src/generate_comfyui.py

2. Evaluate:
   python src/evaluate_clip_dino.py

3. Statistics:
   python src/statistics.py

4. Figures:
   python src/plot_figures.py

## Image counts

Main analysis: 1910 images
Calibration: 40 images (not in main analysis)
Total generated: 1950 images

## Notes

- R² is affine-invariant; min-max normalization cannot change R².
- Perceptual equivalence calibration is required for cross-syntax comparison.
- ComfyUI workflow API JSON should be exported from your local ComfyUI.
