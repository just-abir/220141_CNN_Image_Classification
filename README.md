# 220141 — CNN Image Classification on FashionMNIST

A PyTorch implementation of a Convolutional Neural Network (CNN) trained on the **FashionMNIST** dataset, evaluated on the standard test set, and additionally stress-tested against **10 real-world custom phone photographs** of clothing items to study the generalization gap between clean benchmark data and noisy real-world images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment & Dependencies](#environment--dependencies)
4. [Dataset](#dataset)
5. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
6. [Model Architecture](#model-architecture)
7. [Training Configuration](#training-configuration)
8. [How to Run](#how-to-run)
9. [Results — Standard FashionMNIST Test Set](#results--standard-fashionmnist-test-set)
10. [Results — Custom Phone Photo Dataset](#results--custom-phone-photo-dataset)
11. [Visualizations Produced](#visualizations-produced)
12. [Error Analysis](#error-analysis)
13. [Key Findings & Discussion](#key-findings--discussion)
14. [Limitations](#limitations)
15. [Future Work](#future-work)
16. [Reproducibility Checklist](#reproducibility-checklist)
17. [Author](#author)
18. [License](#license)

---

## Project Overview

This project implements an end-to-end image classification pipeline:

- A custom **2-convolutional-layer CNN** is built from scratch in PyTorch (no pretrained backbone).
- The model is trained and validated on the standard **FashionMNIST** dataset (60,000 training images / 10,000 test images, 10 clothing classes).
- Training/validation loss and accuracy curves are tracked and plotted per epoch.
- A **confusion matrix** heatmap is generated on the full test set to analyze per-class performance.
- The trained model is then evaluated on **10 custom phone-camera photographs** of real clothing items (captured, cropped, and manually labeled by the author) to test out-of-distribution generalization.
- A full **visual error analysis** is performed by sampling misclassified test images and custom images, comparing true vs. predicted labels with confidence scores.
- The trained model weights are serialized as `220141.pth`.

The student/registration ID associated with this project is **220141**, referenced in the trained model filename and the linked dataset repository.

---

## Repository Structure

```
220141_CNN_Image_Classification/
│
├── 220141_CNN_Image_Classification.ipynb   # Main Jupyter/Colab notebook (this file)
├── 220141.pth                              # Saved trained model state_dict (generated after training)
│
├── data/                                   # Auto-downloaded FashionMNIST dataset (torchvision)
│   └── FashionMNIST/
│       └── raw/
│
├── 220141_CNN_Image_Classification/        # Cloned GitHub repo containing custom dataset
│   └── dataset/
│       ├── 1_Shoe.jpeg
│       ├── 02_Shirt.jpg
│       ├── 03_Pant.jpg
│       ├── 04_Jacket.jpg
│       ├── 05_sneaker.jpg
│       ├── 06_pullover.jpg
│       ├── 07_Coat.png
│       ├── 08_Pholo_Shirt.jpg
│       ├── 09_dress.jpg
│       └── 10_Bag.jpeg
│
├── training_history.png                    # Loss & accuracy curves (saved by notebook)
├── confusion_matrix.png                    # Test-set confusion matrix heatmap
├── custom_prediction_gallery.png           # Custom photo predictions gallery
├── error_analysis.png                      # Misclassified sample visualization
│
└── README.md                               # This file
```

> **Note:** The custom dataset images are pulled at runtime via `git clone https://github.com/just-abir/220141_CNN_Image_Classification`, so the `dataset/` folder is not stored locally until the notebook is executed.

---

## Environment & Dependencies

| Package        | Purpose                                   |
|-----------------|--------------------------------------------|
| `torch`         | Core deep learning framework               |
| `torchvision`   | FashionMNIST dataset loader, transforms    |
| `numpy`         | Numerical operations                        |
| `matplotlib`    | Plotting loss/accuracy curves & galleries  |
| `Pillow (PIL)`  | Custom image loading and preprocessing      |
| `scikit-learn`  | Confusion matrix computation                |
| `seaborn`       | Confusion matrix heatmap styling            |

Install everything with:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn seaborn
```

**Hardware:** The notebook auto-detects and uses a CUDA-enabled GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This project was executed with `Using device: cuda`, i.e., on a GPU-backed runtime (e.g., Google Colab GPU instance).

---

## Dataset

### 1. FashionMNIST (Primary Training/Test Data)

- **Source:** `torchvision.datasets.FashionMNIST` (auto-downloaded).
- **Training set:** 60,000 grayscale images, 28×28 pixels.
- **Test set:** 10,000 grayscale images, 28×28 pixels.
- **Classes (10):**

  | Label | Class Name   |
  |-------|--------------|
  | 0     | T-shirt/top  |
  | 1     | Trouser      |
  | 2     | Pullover     |
  | 3     | Dress        |
  | 4     | Coat         |
  | 5     | Sandal       |
  | 6     | Shirt        |
  | 7     | Sneaker      |
  | 8     | Bag          |
  | 9     | Ankle boot   |

### 2. Custom Phone Photo Dataset (Out-of-Distribution Test)

10 real-world photographs of clothing/footwear items taken with a phone camera, hosted in the GitHub repository [`just-abir/220141_CNN_Image_Classification`](https://github.com/just-abir/220141_CNN_Image_Classification) under `dataset/`. Each image was manually assigned a ground-truth label matching the FashionMNIST class taxonomy:

| Filename              | Ground-Truth Label |
|------------------------|---------------------|
| `1_Shoe.jpeg`          | Sneaker             |
| `02_Shirt.jpg`         | Shirt               |
| `03_Pant.jpg`          | Trouser             |
| `04_Jacket.jpg`        | Coat                |
| `05_sneaker.jpg`       | Sneaker             |
| `06_pullover.jpg`      | Pullover            |
| `07_Coat.png`          | Coat                |
| `08_Pholo_Shirt.jpg`   | T-shirt/top         |
| `09_dress.jpg`         | Dress                |
| `10_Bag.jpeg`          | Bag                 |

---

## Data Preprocessing Pipeline

### Standard FashionMNIST transform

```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

- Converts to single-channel grayscale.
- Resizes to the canonical 28×28 FashionMNIST resolution.
- Converts to a PyTorch tensor (values in `[0, 1]`).
- Normalizes with mean = 0.5, std = 0.5, mapping pixel values to `[-1, 1]`.

### Custom photo preprocessing (with auto-invert correction)

Real phone photos have a **light background / dark garment**, which is the *opposite* color convention of FashionMNIST (dark background, light garment silhouette on a black canvas). A naive application of the standard transform therefore produces color-inverted, out-of-distribution inputs. This was diagnosed and fixed with an **auto-invert step**:

```python
def load_custom_image(image_path, transform):
    image = Image.open(image_path).convert('L')      # grayscale
    image = transform(image)                          # resize + tensor (no normalize yet)

    if image.mean() > 0.5:                             # background is bright
        image = 1.0 - image                             # invert colors

    image = (image - 0.5) / 0.5                        # normalize to match training distribution
    image = image.unsqueeze(0)                          # add batch dimension
    return image

custom_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
```

This is one of the more important engineering decisions in the notebook: it directly targets the **domain shift** between clean, pre-processed benchmark images and unconstrained real-world photographs.

---

## Model Architecture

A lightweight custom CNN (`class CNN(nn.Module)`) designed for 28×28 grayscale input:

```
Input: (1, 28, 28)
   │
   ├─ Conv2d(1 → 32, kernel=3, padding=1)  + ReLU
   ├─ MaxPool2d(2, 2)                       → (32, 14, 14)
   │
   ├─ Conv2d(32 → 64, kernel=3, padding=1) + ReLU
   ├─ MaxPool2d(2, 2)                       → (64, 7, 7)
   │
   ├─ Flatten                               → 64 * 7 * 7 = 3136
   ├─ Linear(3136 → 128) + ReLU
   ├─ Dropout(p = 0.25)
   └─ Linear(128 → 10)                      → class logits
```

**Printed model summary:**

```
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU()
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)
```

**Design rationale:**
- Two convolutional blocks are sufficient for 28×28 low-resolution grayscale images; deeper stacks tend to overfit small images without significant gains.
- `Dropout(0.25)` before the final classification layer regularizes the fully-connected head, which has the largest number of parameters in the network.
- ReLU activations throughout for fast, stable convergence.
- MaxPooling halves spatial resolution twice (28→14→7), keeping the flattened feature vector (3136-d) manageable.

---

## Training Configuration

| Hyperparameter        | Value                          |
|-------------------------|--------------------------------|
| Loss function            | `CrossEntropyLoss`             |
| Optimizer                | `Adam`                          |
| Learning rate             | `0.001`                        |
| Batch size (train/test)  | `64`                            |
| Epochs per run            | `10`                            |
| Total epochs executed     | `20` (two sequential 10-epoch training runs continuing from the same weights) |
| Data shuffling             | Enabled for training loader, disabled for test loader |

Training/validation metrics (loss and accuracy) are tracked every epoch and stored in:
```python
train_loss_history, train_acc_history, val_loss_history, val_acc_history
```

The trained weights are saved with:
```python
torch.save(model.state_dict(), '220141.pth')
```

---

## How to Run

1. **Clone this repository / open the notebook** in Jupyter, Google Colab, or VS Code.
2. **Install dependencies** (see [Environment & Dependencies](#environment--dependencies)).
3. **Run all cells top-to-bottom.** The notebook will automatically:
   - Download FashionMNIST via `torchvision`.
   - Clone the custom photo dataset repository.
   - Build, train, and validate the CNN.
   - Save the trained model as `220141.pth`.
   - Generate and save all plots (`training_history.png`, `confusion_matrix.png`, `custom_prediction_gallery.png`, `error_analysis.png`).
4. **Inspect the printed epoch-by-epoch logs** and the generated figures for results.

To run on GPU, ensure a CUDA-capable runtime is selected (e.g., in Colab: *Runtime → Change runtime type → GPU*).

---

## Results — Standard FashionMNIST Test Set

### Training curve (Run 1, epochs 1–10)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1  | 0.4741 | 82.96% | 0.3292 | 88.04% |
| 2  | 0.3093 | 88.65% | 0.2852 | 89.55% |
| 3  | 0.2650 | 90.26% | 0.2740 | 90.04% |
| 4  | 0.2329 | 91.26% | 0.2521 | 91.08% |
| 5  | 0.2080 | 92.34% | 0.2424 | 91.58% |
| 6  | 0.1878 | 93.04% | 0.2397 | 91.56% |
| 7  | 0.1670 | 93.87% | 0.2271 | 91.99% |
| 8  | 0.1494 | 94.31% | 0.2425 | 91.94% |
| 9  | 0.1369 | 94.79% | 0.2464 | 92.13% |
| 10 | 0.1230 | 95.26% | 0.2494 | 92.21% |

### Continued training (Run 2, epochs 11–20 — resumed from Run 1 weights)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 11 | 0.1106 | 95.82% | 0.2548 | 92.21% |
| 12 | 0.0994 | 96.32% | 0.2592 | 92.28% |
| 13 | 0.0905 | 96.60% | 0.2811 | 91.86% |
| 14 | 0.0823 | 96.88% | 0.3179 | 92.03% |
| 15 | 0.0757 | 97.12% | 0.3246 | 92.24% |
| 16 | 0.0707 | 97.27% | 0.3307 | 92.43% |
| 17 | 0.0683 | 97.31% | 0.3208 | 92.08% |
| 18 | 0.0600 | 97.62% | 0.3512 | 92.22% |
| 19 | 0.0585 | 97.75% | 0.3545 | 91.79% |
| 20 | 0.0536 | 97.97% | 0.3713 | 91.99% |

**Summary:**
- **Best validation accuracy:** ~92.43% (epoch 16).
- **Final training accuracy:** 97.97% — the training loss continues to fall steadily.
- **Validation loss begins rising after ~epoch 10–11** while training loss keeps dropping — a clear **overfitting** signature once total training exceeds ~10–12 epochs.
- The gap between train accuracy (97.97%) and validation accuracy (91.99%) at epoch 20 (~6 points) confirms the model has begun memorizing training-set-specific patterns.

### Confusion Matrix

A 10×10 confusion matrix (`confusion_matrix.png`) was computed over the full 10,000-image test set using `sklearn.metrics.confusion_matrix` and visualized as a Seaborn heatmap. The classes most frequently confused with one another are the visually similar upper-body garments — **Shirt, T-shirt/top, Pullover, and Coat** — which is a well-documented characteristic of FashionMNIST, since these four classes share very similar silhouettes at 28×28 resolution.

---

## Results — Custom Phone Photo Dataset

After fixing the color-inversion preprocessing bug (Step 8.4), the model produced the following predictions on the 10 real-world custom photos:

| Image | True Label | Predicted Label | Confidence | Correct? |
|-------|------------|------------------|------------|----------|
| `1_Shoe.jpeg` | Sneaker | Ankle boot | 99.8% | ❌ |
| `02_Shirt.jpg` | Shirt | Shirt | 100.0% | ✅ |
| `03_Pant.jpg` | Trouser | Trouser | 96.5% | ✅ |
| `04_Jacket.jpg` | Coat | Shirt | 91.3% | ❌ |
| `05_sneaker.jpg` | Sneaker | Bag | 100.0% | ❌ |
| `06_pullover.jpg` | Pullover | Coat | 68.3% | ❌ |
| `07_Coat.png` | Coat | Coat | 97.8% | ✅ |
| `08_Pholo_Shirt.jpg` | T-shirt/top | T-shirt/top | 99.0% | ✅ |
| `09_dress.jpg` | Dress | Bag | 100.0% | ❌ |
| `10_Bag.jpeg` | Bag | Coat | 40.1% | ❌ |

**Custom-set accuracy: 4/10 = 40%**, dramatically lower than the ~92% test accuracy achieved on the standard FashionMNIST test set.

**Notable observations:**
- Several *incorrect* predictions carry **very high confidence** (e.g., `1_Shoe.jpeg` at 99.8% and `05_sneaker.jpg` at 100.0%), meaning the model is not just wrong but **confidently wrong** — a hallmark of distribution shift rather than genuine class ambiguity.
- The single *lowest-confidence* prediction (`10_Bag.jpeg`, 40.1%) is also incorrect, suggesting the model was at least appropriately uncertain in that case.
- Correct predictions cluster around images with strong silhouette contrast and simple, centered garments (`Shirt`, `Trouser`, `Coat`, `T-shirt/top`).

---

## Visualizations Produced

The notebook generates and saves the following figures:

1. **`training_history.png`** — Side-by-side line plots of training/validation loss and training/validation accuracy across all 20 epochs.
2. **`confusion_matrix.png`** — 10×10 annotated heatmap of predicted vs. true labels on the FashionMNIST test set.
3. **Model input visualization** (`plt.suptitle("What the Model Actually Sees...")`) — Displays the 10 custom images exactly as fed into the network (28×28 grayscale, inverted/normalized), useful for debugging preprocessing.
4. **Reference gallery** — 5 sample FashionMNIST training images with their class labels, for visual comparison against the custom photos.
5. **`custom_prediction_gallery.png`** — A 2×5 grid of the original custom RGB photos with true/predicted label overlays, color-coded **green** for correct and **red** for incorrect predictions.
6. **`error_analysis.png`** — 3 randomly sampled misclassified test-set images with true vs. predicted label annotations.

---

## Error Analysis

Out of the 10,000-image standard test set, the model collected at least **51 misclassified samples** (collection was capped early at 50+ for efficiency). Random inspection of 3 misclassified samples (Step 11) is used to qualitatively audit failure modes, complementing the quantitative confusion matrix.

Common failure patterns observed across both the standard test set and the custom photo set:

- **Semantic overlap among upper-body garments** (Shirt ↔ T-shirt/top ↔ Pullover ↔ Coat) is the dominant source of error.
- **Silhouette-dependent footwear confusion** (Sneaker ↔ Ankle boot ↔ Sandal) appears when the item's profile is ambiguous or partially cropped.
- **Domain shift on custom photos**: real-world lighting, shadows, folds, background texture, and non-centered framing significantly increase error rate relative to the clean, centered, background-free FashionMNIST images.

---

## Key Findings & Discussion

1. **Simple 2-conv-layer CNNs are highly effective on FashionMNIST**, reaching ~92% validation accuracy with a compact ~1.6M-parameter architecture and only 10–16 epochs of training.
2. **Overfitting emerges quickly** once training continues past ~10–12 epochs: training accuracy keeps climbing toward 98% while validation accuracy plateaus around 92% and validation loss starts increasing. This suggests **early stopping around epoch 10–16** or **stronger regularization** (e.g., higher dropout, weight decay, data augmentation) would improve generalization.
3. **Benchmark accuracy does not guarantee real-world robustness.** The drop from ~92% (clean test set) to 40% (custom photos) is a striking demonstration of the **domain gap** between curated benchmark datasets and unconstrained real-world inputs — a critical lesson in deploying vision models beyond their training distribution.
4. **Preprocessing assumptions matter enormously.** The single auto-invert fix (Step 8.4) was necessary just to get the custom images into a format the model could meaningfully interpret, underscoring how sensitive small CNNs are to pixel-value conventions they were never exposed to during training.
5. **High-confidence errors on out-of-distribution data** highlight a broader concern in deep learning: softmax confidence is not a reliable proxy for correctness under distribution shift.

---

## Limitations

- The custom evaluation set is very small (n = 10), so the 40% figure is a directional signal, not a statistically robust estimate.
- No data augmentation (rotation, crop, color jitter, elastic distortion) was applied during training, which likely would have improved robustness to the custom photos.
- No learning-rate scheduling, early stopping, or checkpoint selection was used — the final-epoch weights (not the best-validation-epoch weights) are what get saved.
- The model was not fine-tuned or calibrated on any real-world photos.
- Only global auto-invert preprocessing was applied to custom images; no object detection/cropping/background removal was performed, so backgrounds and framing still differ from FashionMNIST's clean canvas.

---

## Future Work

- Add **data augmentation** (random rotation, translation, and slight color/brightness jitter) to the training pipeline to close the real-world domain gap.
- Introduce **early stopping** based on validation loss to avoid the overfitting observed after epoch ~10–16.
- Add **batch normalization** layers after each convolution to stabilize and potentially accelerate training.
- Expand the custom photo dataset to more samples per class for a statistically meaningful out-of-distribution benchmark.
- Explore **transfer learning** from a model pretrained on a broader natural-image dataset, then fine-tune on FashionMNIST + custom photos jointly.
- Apply **background segmentation/cropping** to custom photos before inference, to better match the clean, centered FashionMNIST image convention.
- Track **per-class precision/recall/F1** in addition to the confusion matrix for a more granular performance breakdown.

---

## Reproducibility Checklist

- [x] Fixed batch size and known optimizer/learning-rate settings documented.
- [x] Dataset source (`torchvision.datasets.FashionMNIST`) and custom dataset (public GitHub repo) both referenced.
- [x] Model architecture fully specified (layer types, sizes, activation functions).
- [x] Training/validation metrics logged per epoch.
- [x] Trained weights checkpoint saved (`220141.pth`).
- [ ] Random seed not explicitly fixed in the notebook — for full reproducibility, set `torch.manual_seed(...)`, `numpy.random.seed(...)`, and `random.seed(...)` before training.

---

## Author

**Student/Registration ID:** 220141
**Project:** CNN Image Classification — FashionMNIST + Custom Phone Photo Evaluation
**Custom dataset repository:** [`github.com/just-abir/220141_CNN_Image_Classification`](https://github.com/just-abir/220141_CNN_Image_Classification)

---

## License

This project is provided for academic/educational purposes. FashionMNIST is released by Zalando Research under the MIT License. Custom photographs are the property of the author and included here solely for coursework evaluation purposes.
