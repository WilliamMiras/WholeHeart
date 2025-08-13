# WholeHeart

## Overview

WholeHeart is a deep learning framework for automated analysis of echocardiogram videos, designed to predict ejection fraction (EF) and perform cardiac structure segmentation. It utilizes a hybrid neural architecture combining ConvNeXt (for spatial feature extraction) and Mamba SSM (for temporal modeling), enabling robust spatiotemporal learning from medical video data. The project is built with PyTorch and supports GPU acceleration for efficient training and inference.

## Features

- Hybrid ConvNeXt + Mamba SSM model for video-based EF prediction and segmentation
- Custom PyTorch dataset loader for the EchoNet dataset
- Automated checkpointing and resume functionality during training
- Training and evaluation scripts
- GPU support

## Dataset

WholeHeart uses the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset, which should be placed in the `echonet_data/` directory with the following structure:

```
echonet_data/
	 FileList.csv
	 VolumeTracings.csv
	 Videos/
		  <video files>.avi
```

## Installation

1. Clone the repository:
	```powershell
	git clone https://github.com/WilliamMiras/WholeHeart.git
	cd WholeHeart
	```
2. Install dependencies (recommended: Python 3.10+):
	```powershell
	pip install -r requirements.txt
	```
3. Download and extract the EchoNet dataset into `echonet_data/` as shown above.

## Model Architecture

The core model (`EchoNetConvMamba`) consists of:
- **ConvNeXt-Tiny** backbone (pretrained, classifier removed)
- **Mamba SSM** layers for temporal modeling
- **EF Head**: Fully connected layers for ejection fraction regression
- **Segmentation Head**: Transposed convolution layers for pixel-wise segmentation

### Training

Run the training script:
```powershell
python train.py
```
This will train the model on the EchoNet train split, automatically saving checkpoints in the `checkpoints/` directory. Training resumes from the latest checkpoint if available.

### Evaluation / Testing

You can run basic CUDA and environment tests with:
```powershell
python test.py
```
For model evaluation, implement or extend the evaluation logic in `train.py` or create a dedicated script.

## File Structure

- `model.py`: Model definition (ConvNeXt + Mamba SSM)
- `dataset.py`: Custom PyTorch dataset for EchoNet
- `train.py`: Training loop, checkpointing, data pipeline
- `test.py`: Environment and CUDA checks
- `requirements.txt`: Python dependencies
- `checkpoints/`: Saved model checkpoints
- `echonet_data/`: Dataset folder

## Troubleshooting

- Ensure your GPU drivers and CUDA are properly installed.
- If you encounter out-of-memory errors, reduce the batch size in `train.py`.
- Dataset files must match the expected structure and filenames.

## License

This project is licensed under the MIT License.