
# Flower Classification and Semantic Segmentation

**Computer Vision | Deep Learning | MATLAB**

A computer vision project investigating two fundamental deep learning tasks: **multi-class image classification** and **semantic segmentation**.

The project explores how neural networks can learn meaningful visual representations, from recognising flower species in an image to identifying individual object pixels. It combines a custom Convolutional Neural Network (CNN), developed and trained from scratch, with an investigation into semantic segmentation using DeepLabv3+ and transfer learning.

The repository contains the MATLAB classification implementation and an accompanying academic project paper documenting the research, methodology, experiments, results and evaluation.

## Project Overview

| Task | Approach | Objective |
|------|----------|-----------|
| Flower Classification | Custom CNN trained from scratch | Classify images into 17 flower categories |
| Semantic Segmentation | DeepLabv3+ and transfer learning | Identify daffodils at the pixel level |

**Technologies:** MATLAB, Deep Learning Toolbox, Image Processing Toolbox, CNNs, DeepLabv3+, Transfer Learning

## 1. Flower Image Classification

### Objective

The first experiment focuses on designing and training a Convolutional Neural Network capable of distinguishing between 17 different flower species.

Rather than using an existing pre-trained classification network, a custom CNN was implemented to investigate feature extraction, network architecture, data augmentation, regularisation and optimisation.

### Dataset

The classification experiment uses the [Oxford 17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), developed by the Visual Geometry Group at the University of Oxford.

- **17 flower categories**
- **80 images per category**
- **1,360 images in total**
- Variations in lighting, background, orientation and scale

### Data Preprocessing

The classification pipeline prepares the dataset through:

1. **Dataset organisation:** Images are organised into labelled folders corresponding to their flower categories.
2. **Image resizing:** Images are resized to 256 × 256 pixels.
3. **Dataset splitting:** Images are randomly divided into 80% training and 20% validation sets.
4. **Data augmentation:** Random transformations increase variation within the training data.

Augmentation includes rotation, translation, scaling, shearing and horizontal reflection.

These transformations aim to improve generalisation and reduce sensitivity to variations in image positioning and appearance.

### CNN Architecture

The custom network consists of four convolutional blocks followed by fully connected classification layers.

| Layer | Configuration |
|-------|---------------|
| Input | 256 × 256 × 3 RGB |
| Convolution Block 1 | 32 filters, 5 × 5 kernel, Batch Normalisation, ReLU, Max Pooling |
| Convolution Block 2 | 64 filters, 3 × 3 kernel, Batch Normalisation, ReLU, Max Pooling |
| Convolution Block 3 | 128 filters, 3 × 3 kernel, Batch Normalisation, ReLU, Max Pooling |
| Convolution Block 4 | 256 filters, 3 × 3 kernel, Batch Normalisation, ReLU, Max Pooling |
| Fully Connected 1 | 512 neurons, ReLU, Dropout (0.5) |
| Fully Connected 2 | 128 neurons, ReLU, Dropout (0.4) |
| Output | 17 neurons, Softmax, Classification |

The convolutional blocks progressively extract visual features, from basic edges and textures to higher-level representations.

Batch normalisation helps stabilise training, while dropout is used to reduce overfitting.

### Training

The CNN is trained using Stochastic Gradient Descent with Momentum (SGDM).

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | SGDM |
| Initial Learning Rate | 0.001 |
| Epochs | 18 |
| Mini-batch Size | 32 |
| Training/Validation Split | 80/20 |
| Validation Frequency | 30 iterations |

### Evaluation

The model is evaluated on the validation dataset using:

- **Classification accuracy:** Measures the percentage of correctly classified images.
- **Confusion matrix:** Identifies categories that are correctly classified or frequently confused.

The MATLAB script also saves the trained network to `classnet.mat` for future use.

Further experimental analysis and discussion are included in the accompanying project paper.

---

## 2. Semantic Segmentation

### Objective

The second experiment explores semantic segmentation, a computer vision task that assigns a class label to every pixel in an image.

Unlike classification, which produces a single category prediction, segmentation identifies the spatial regions belonging to an object.

The objective is to distinguish daffodil pixels from their surrounding background.

### Approach

The experiment investigates **DeepLabv3+**, a semantic segmentation architecture that combines multi-scale feature extraction with an encoder-decoder structure.

Key architectural concepts include:

- **Encoder:** Extracts high-level semantic features from images.
- **Atrous convolution:** Captures information over larger receptive fields without necessarily reducing spatial resolution.
- **Atrous Spatial Pyramid Pooling (ASPP):** Combines features captured at different spatial scales.
- **Decoder:** Refines predictions and helps recover object boundaries.

The experiment explores transfer learning to adapt a pre-trained network to the specific task of segmenting daffodils.

### Evaluation

Semantic segmentation can be evaluated using metrics including:

- **Global Accuracy:** The proportion of correctly classified pixels across the dataset.
- **Intersection over Union (IoU):** Measures overlap between predicted segmentation masks and ground-truth labels.
- **Mean Intersection over Union (Mean IoU):** Averages IoU across the evaluated classes.

The accompanying academic paper contains the experimental methodology, quantitative results, visual comparisons and discussion of model performance.

**Note:** The segmentation implementation is not included in this repository. The experiment is documented in the accompanying project paper.

---

## 3. Installation and Usage

### Requirements

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Oxford 17 Category Flower Dataset

### Clone the Repository

```bash
git clone https://github.com/samiVanga/Computer_Vision-classification-and-Segmentation.git
cd Computer_Vision-classification-and-Segmentation
```

### Download the Dataset

Download the [Oxford 17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and extract the images into a directory named `17flowers`.

Example directory structure:

```text
Computer_Vision-classification-and-Segmentation/
│
├── README.md
├── classification.m
├── [project paper PDF]
│
└── 17flowers/
    ├── image_0001.jpg
    ├── image_0002.jpg
    └── ...
```

The `17flowers` directory must be created using the downloaded images. The dataset is not included in the repository.

### Run Classification

Open MATLAB and set the repository directory as your current working folder.

Run:

```matlab
run('classification.m')
```

The script:

1. Organises the flower images into labelled folders.
2. Preprocesses and augments the dataset.
3. Creates the training and validation partitions.
4. Constructs the CNN.
5. Trains the network.
6. Evaluates validation accuracy.
7. Generates a confusion matrix.
8. Saves the trained model.

To load the saved network after training:

```matlab
load('classnet.mat')
```

**Note:** The classification script uses relative paths based on MATLAB's current working directory. Run the script from the repository root.

---

## 4. Project Paper

An accompanying academic paper is included in this repository, providing a more detailed discussion of the research and experimental work.

The paper covers:

- Background and motivation for image classification and semantic segmentation.
- Deep learning architectures and design decisions.
- Dataset preparation and preprocessing.
- Model training and optimisation.
- Experimental evaluation and performance metrics.
- Analysis of results and model limitations.

Refer to the PDF in the repository for the full methodology, quantitative findings and conclusions.

---

## 5. Key Learning Outcomes

This project provided practical experience in:

- **CNN development:** Designing and training a deep learning network from scratch.
- **Image preprocessing:** Preparing and organising visual datasets for supervised learning.
- **Data augmentation:** Improving training diversity through geometric transformations.
- **Model optimisation:** Investigating training configurations and regularisation techniques.
- **Transfer learning:** Exploring how pre-trained representations can support semantic segmentation.
- **Model evaluation:** Analysing classification and segmentation performance using task-specific metrics.

The project also highlights the differences between image-level recognition and pixel-level prediction, particularly the additional challenges of precise object localisation and boundary detection.

---

## 6. Future Improvements

Potential extensions include:

- Publishing the semantic segmentation implementation.
- Comparing the custom CNN against pre-trained classification architectures.
- Evaluating generalisation using a separate held-out test dataset.
- Investigating additional augmentation and regularisation techniques.
- Including example predictions and segmentation masks.
- Exploring alternative segmentation architectures.

---

## References

**Oxford 17 Category Flower Dataset**

Nilsback, M.-E. and Zisserman, A. (2006). *A Visual Vocabulary for Flower Classification*. IEEE Conference on Computer Vision and Pattern Recognition.

[Dataset and documentation](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

**DeepLabv3+**

Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F. and Adam, H. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. European Conference on Computer Vision.

[Research paper](https://arxiv.org/abs/1802.02611)

---

## Author

**Samiksha Vanga**

BSc Computer Science, University of Nottingham

[GitHub](https://github.com/samiVanga)
