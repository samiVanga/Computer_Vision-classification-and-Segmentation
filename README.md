# Flower Classification with Custom CNNs and Daffodil Segmentation via Transfer Learning

This repository contains a deep learning project that performs:

1. **Multi-class flower classification** using a custom Convolutional Neural Network (CNN) trained from scratch on the Oxford 17 Flower Dataset.
2. **Semantic segmentation of daffodils** using a pre-trained DeepLabv3+ model with a ResNet-50 backbone to isolate flowers from complex backgrounds.

Developed using MATLAB.

---

## Models Used

### Classification (Custom CNN)
- Input: 256x256 RGB flower images
- Architecture: 5 convolutional blocks + 3 dense layers
- Data: Oxford 17 Flower Dataset
- Achieved Accuracy: **73.90%**
- Features: BatchNorm, ReLU, MaxPooling, Dropout, SGD optimizer

### Segmentation (DeepLabv3+ with ResNet-50)
- Input: 256x256 RGB daffodil images
- Pretrained on: Deeplabv3+ using RestNet
- Training: Fine-tuned on custom binary-labeled daffodil dataset
- Achieved Accuracy: **92.21% IoU**, **96.93% global accuracy**

---


