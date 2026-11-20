# SAM 3 with LoRA for metal subtype segmentation (Text-Full-D)

Code for the proposed configuration (Text-Full-D) in:
Ranasinghe, D., Sirimewan, D., Bazli, M., Arashpour, M. (under review). Waste Management.

SAM 3 is adapted with module-specific LoRA (vision encoder r = 4, DETR encoder r = 16,
DETR decoder r = 32, mask decoder r = 4) and prompted with one class name per forward pass.

## Files
- `train.py`: training
- `evaluate.py`: evaluation (metrics as in Section 2.5 of the article)
- `model.py`: SAM 3 with LoRA
- `dataloader.py`: COCO dataset loader
- `utils.py`: helper functions
- `config.yaml`: configuration used for the reported results

## Requirements
- Python 3.10 or later and an NVIDIA GPU
- `pip install -r requirements.txt`
- Access to SAM 3 on Hugging Face (`facebook/sam3`). Log in with `huggingface-cli login`.

## Data
The test split of WasteMetal-6 is available at https://doi.org/10.17632/ch34f38kz5.2.
The full dataset is available on request.

Set `dataset.root` in `config.yaml`. Each split folder contains its images and `_annotations.coco.json`:
```
<dataset.root>/
├── train/
├── valid/
└── test/
```

## Train
```
python train.py --config config.yaml
```

## Evaluate
```
python evaluate.py --config config.yaml \
  --checkpoint outputs/text_full_d/checkpoints/best.pt \
  --split test --results-dir outputs/text_full_d/results \
  --score-threshold 0.15 --mask-threshold 0.50 --nms-iou-threshold 0.50
```
Add `--save-qualitative` to save side-by-side images of ground truth and predictions.

## License
This code is released under the MIT License. SAM 3 weights are subject to Meta's SAM license and are not included..
