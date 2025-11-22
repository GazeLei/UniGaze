# UniGaze Easy Loader

A tiny, dependency-light **Python package** to load **UniGaze** pretrained models from [Hugging Face](https://huggingface.co/UniGaze/UniGaze-models/tree/main).



## 📦 Installation

> Install a matching PyTorch first.

** CUDA 12.8 example**
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install unigaze
```



## 🚀 Quick Start

```python
import torch
model = unigaze.load("unigaze_h14_joint", device="cuda")   # downloads weights from HF on first use
# Input: normalized batch (B, 3, 224, 224)
image_normalized_batch = torch.ones((10, 3, 224, 224), device="cuda")
# Output: {'pred_gaze': (B, 2)} with (pitch, yaw)
pred_gaze = model(image_normalized_batch)['pred_gaze']
print(pred_gaze.shape)  # torch.Size([10, 2])
```