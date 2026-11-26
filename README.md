<p align="center"><img src="docs/assets/cover.svg" alt="CoRE-Net cover" width="100%"></p>

# CoRE-Net

**Region Matching and Prior-Guided Contour Refinement for Referring Remote Sensing Image Segmentation**

CoRE-Net segments a target in a remote sensing image from a natural-language description. Similar objects may match different parts of the description, and nearby edges may interfere with the final mask. The model uses semantic roles and visual evidence to match candidate regions against the full expression, then uses the resulting target prior to guide contour refinement. The implementation supports **RefSegRS** and **RRSIS-D**.

This repository contains the source code for the paper's model. Dataset files, pretrained encoder weights, and trained checkpoints are not included.

## Setting up

Clone the repository and enter its root directory:

```bash
git clone https://github.com/Hike-Curry/CoRE-Net.git
cd CoRE-Net
```

Create a Python 3.10 environment and install a PyTorch build compatible with your CUDA version. The project declares PyTorch 2.3 or newer and the remaining packages in `requirements.txt`.

```bash
conda create -n core-net python=3.10
conda activate core-net
# Install PyTorch and torchvision for your CUDA version first.
pip install -r requirements.txt
python -m nltk.downloader -d nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

Place generic Swin-B classification weights and BERT-base-uncased files at these paths, or pass their locations through the command-line arguments:

```text
pretrained_weights/swin_base_patch4.pth
bert_base_uncased/config.json
bert_base_uncased/vocab.txt
bert_base_uncased/pytorch_model.bin
```

The supplied archive does not include a clean-environment installation log, so the exact Python, CUDA, and package combination has not been independently verified for this release.

## Datasets

Prepare the datasets in the layout expected by the readers:

```text
data/
  refsegrs/
    output_phrase_train.txt
    output_phrase_val.txt
    output_phrase_test.txt
    images/
    masks/
  rrsisd/
    refs(unc).p
    instances.json
    images/
```

Use `--data-root` if your dataset is elsewhere. Obtain each dataset under its own distribution terms.

## Training

Both datasets use the same `train.py` entry. Select the dataset with `--dataset`:

```bash
python -u train.py --dataset refsegrs --epochs 150 --batch-size 8 --output outputs/refsegrs
python -u train.py --dataset rrsisd --epochs 100 --batch-size 8 --output outputs/rrsisd
```

Training reports training and validation metrics. The checkpoint with the highest **validation mIoU** is saved as `best/best.pth` under the selected output directory. Test data is not used for checkpoint selection.

| Setting | RefSegRS | RRSIS-D |
| --- | ---: | ---: |
| Epochs | 150 | 100 |
| Visual encoder and fusion learning rate | 3e-5 | 2e-6 |
| Text encoder learning rate | 1e-5 | 1e-6 |
| Projection and decoder learning rate | 1e-4 | 1e-5 |
| Maximum text tokens | 20 | 22 |

Both configurations use AdamW, weight decay 0.01, batch size 8, and 480 × 480 inputs. See `args.py` and `engine.py` for the remaining training settings.

## Testing

Run testing with a checkpoint trained for the matching dataset:

```bash
python test.py --dataset refsegrs --checkpoint outputs/refsegrs/best/best.pth
python test.py --dataset rrsisd --checkpoint outputs/rrsisd/best/best.pth
```

Testing reports mIoU, oIoU, and precision at IoU thresholds 0.5–0.9. Use `--device cpu` when evaluating without a GPU. Checkpoint files are not included in this source release.

## Code layout

| Path | Purpose |
| --- | --- |
| `train.py`, `test.py` | Training and standalone test entry points |
| `args.py` | Shared dataset and training configuration |
| `lib/` | Vision backbone, region matching, and contour refinement |
| `data/` | RefSegRS and RRSIS-D readers and language processing |
| `loss/` | Segmentation and candidate-quality losses |
| `engine.py` | Training, validation, and checkpoint selection |

## Citation and acknowledgements

The paper's citation record will be added when available. Third-party code notices and license texts are in `THIRD_PARTY_NOTICES.txt` and `LICENSES/`.
