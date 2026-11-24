<p align="center"><img src="docs/assets/cover.svg" alt="Language-grounded remote sensing segmentation" width="100%"></p>

# Language-Grounded Remote Sensing Segmentation

**Describe an object. Locate it from above.**

基于自然语言描述，从遥感图像中分割目标区域。本项目基于 FIANet 代码开展研究，当前实现为 language-grounded V1 改进模型，并非原论文未经修改的基线。

This research project segments objects in remote sensing images from natural-language descriptions. Built on the FIANet codebase, it contains a modified language-grounded V1 model rather than the unmodified original implementation.

## Overview

遥感图像中常有多个外观相似的目标，仅识别类别不足以确定描述所指的对象。当前模型将文本信息引入视觉特征学习，结合候选区域预测与细节细化，利用描述中的属性和空间线索定位目标。

Remote sensing images often contain several objects with similar appearances. The model combines language-conditioned visual features, candidate mask prediction, and detail refinement to support segmentation from descriptive and spatial cues.

- **Two datasets:** RRSIS-D and RefSegRS.
- **One training entry:** choose the dataset with `--dataset`.
- **Separate evaluation:** select checkpoints on validation data and report test metrics separately.

The cover is a conceptual illustration, not a model prediction.

The repository currently contains the project introduction and cover. Source code and reproducibility materials are being prepared for release.

## Preparation

Dependencies are declared in `requirements.txt` in the local research code. They include legacy MMCV and tokenizer versions; a clean installation and a tested Python/PyTorch/CUDA compatibility matrix remain to be verified before release.

Prepare the selected dataset, BERT files in `bert_base_uncased/`, and a compatible window-12 generic Swin checkpoint. Dataset files and pretrained weights are not bundled with source code. Detailed dataset preparation instructions are pending.

## Training

Both datasets share the same entry. Select checkpoints using validation data:

```bash
python -u train.py --dataset rrsisd --data-root data --vision-init pretrained_weights/swin_base_patch4.pth --select-best-by val --epochs 200 --batch-size 8 --workers 4 --output outputs/rrsisd_release

python -u train.py --dataset refsegrs --data-root data/refsegrs --vision-init pretrained_weights/swin_base_patch4.pth --select-best-by val --epochs 200 --batch-size 8 --workers 4 --output outputs/refsegrs_release
```

`--warm-start` loads compatible weights for a new run without restoring the optimizer. Historical test-selected checkpoints remain development-derived and must not be presented as unbiased held-out evidence.

## Evaluation

```bash
python -u test.py --dataset rrsisd --data-root data --checkpoint model_best/rrsisd_best.pth --device cuda --workers 4
python -u test.py --dataset refsegrs --data-root data/refsegrs --checkpoint model_best/refsegrs_best.pth --device cuda --workers 4
```

Supply matching checkpoints yourself; these paths do not imply that weights are published. Without `--device cuda`, testing defaults to CPU. Checkpoint metadata must match the dataset and supported architecture.

## Code structure

| Path | Purpose |
| --- | --- |
| `train.py` / `test.py` | Training and standalone evaluation |
| `args.py` | Command-line configuration |
| `lib/grounding.py` | Current model |
| `data/grounding.py` | Shared dataset interface |
| `loss/grounding.py` | Mask and candidate-quality objectives |
| `engine.py` | Training and evaluation orchestration |
| `training_utils.py` | Metrics and experiment artifacts |

## Acknowledgements and release status

This project builds on **Exploring Fine-Grained Image-Text Alignment for Referring Remote Sensing Image Segmentation** and its FIANet implementation. Credit also belongs to the datasets and upstream model implementations used by the code.

This is a release preparation draft. The final project name, citation metadata, upstream links, dataset guide, verified environment, and redistribution license remain to be completed. No new license grant is made by this README; applicable upstream licenses and notices must be retained. No benchmark reproduction is claimed by this documentation update.