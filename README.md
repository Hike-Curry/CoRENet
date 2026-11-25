<p align="center"><img src="docs/assets/cover.svg" alt="Language-grounded remote sensing segmentation — conceptual illustration" width="100%"></p>

# Language-Grounded Remote Sensing Segmentation

This repository presents our research on referring remote sensing image segmentation, built on [FIANet](https://github.com/Shaosifan/FIANet). Given an image and a natural-language description, the model predicts the region corresponding to the described object. The current implementation combines language-conditioned visual features, candidate masks, and detail refinement, and supports RRSIS-D and RefSegRS.

> **Release status:** Project introduction and cover are available. Source code, pretrained weights, and complete setup instructions are being prepared. The commands below describe the local implementation and require the forthcoming code release.

## Setting Up

### Preliminaries

The implementation uses PyTorch, Swin Transformer, and BERT. The tested environment will be documented with the code release. Current argument parsing requires Python 3.9 or later because it uses `argparse.BooleanOptionalAction`; this is a language-version requirement, not a verified dependency configuration.

### Package Dependencies

Once the source code is available, clone this repository and enter its root directory:

```bash
git clone https://github.com/Hike-Curry/core.git
cd core
```

Install a compatible PyTorch/CUDA environment and the packages specified in `requirements.txt`. The dependency list contains legacy packages and still requires clean-environment verification; installation instructions will be finalized after that check.

### Data and Pretrained Models

Prepare RRSIS-D or RefSegRS and place the required files at the paths passed to `--data-root`. The default local layout is:

```text
data/                          # RRSIS-D root
data/refsegrs/                 # RefSegRS root
bert_base_uncased/             # BERT model and tokenizer files
pretrained_weights/           # Generic Swin weights
model_best/                   # Dataset-specific evaluation checkpoints
```

Training requires compatible window-12 Swin initialization and BERT files. Dataset download links, exact directory layouts, and model download instructions will accompany the code release.

## Training

For RRSIS-D:

```bash
python -u train.py --dataset rrsisd --data-root data --vision-init pretrained_weights/swin_base_patch4.pth --select-best-by val --epochs 200 --batch-size 8 --workers 4 --output outputs/rrsisd_release
```

For RefSegRS:

```bash
python -u train.py --dataset refsegrs --data-root data/refsegrs --vision-init pretrained_weights/swin_base_patch4.pth --select-best-by val --epochs 200 --batch-size 8 --workers 4 --output outputs/refsegrs_release
```

Both datasets share one training entry. Checkpoints are selected using validation mIoU, with test metrics reported separately. Historical test-selected runs are development experiments and are not unbiased held-out evaluations.

## Testing

For RefSegRS:

```bash
python -u test.py --dataset refsegrs --data-root data/refsegrs --checkpoint model_best/refsegrs_best.pth --device cuda --workers 4
```

For RRSIS-D:

```bash
python -u test.py --dataset rrsisd --data-root data --checkpoint model_best/rrsisd_best.pth --device cuda --workers 4
```

Replace the checkpoint paths with compatible trained weights. Model weights are not currently included. Testing defaults to CPU when `--device cuda` is omitted.

## Citation

Citation information for this project will be added when available. For the original FIANet work, please refer to its [paper and citation information](https://github.com/Shaosifan/FIANet#citation).

## Acknowledgements

Our code builds on [FIANet](https://github.com/Shaosifan/FIANet), the implementation of *Exploring Fine-Grained Image-Text Alignment for Referring Remote Sensing Image Segmentation*. We thank its authors and the dataset contributors for sharing their work. This repository contains a modified model, not the unmodified FIANet baseline.

Redistribution licensing and upstream notices will be finalized before public code release.
