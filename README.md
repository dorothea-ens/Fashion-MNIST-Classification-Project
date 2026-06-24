# Fashion-MNIST Classification: Benchmarking Deep Learning Against Ensemble Methods

## Project Overview

This project presents a comparative analysis of two fundamentally different machine learning paradigms for fashion product recognition using the Fashion-MNIST dataset. The study pits a **spatial feature extraction approach** (Convolutional Neural Network) against a **probabilistic ensemble method** (Random Forest) to determine which strategy offers superior performance for automated clothing categorization.

### Research Questions Addressed

- How does spatial feature learning compare to pixel-wise independence in image classification?
- Can a traditional ensemble method compete with deep learning on structured visual data?
- What is the practical trade-off between computational efficiency and predictive accuracy?

### Key Results at a Glance

| Aspect | Winner | Value |
|--------|--------|-------|
| Test Accuracy | CNN | 91.03% |
| Training Speed | Random Forest | 11.1 seconds |
| Generalization | CNN | Small train-test gap |
| Production Suitability | CNN | Recommended |

---

## Dataset Description

### The Fashion-MNIST Benchmark

Fashion-MNIST serves as a modern replacement for the classic MNIST digit dataset, designed specifically for evaluating image classification algorithms in a realistic fashion context. The dataset comprises:

| Attribute | Specification |
|-----------|---------------|
| Total Samples | 70,000 grayscale images |
| Training Samples | 60,000 (further split: 48,000 train + 12,000 validation) |
| Test Samples | 10,000 |
| Resolution | 28 × 28 pixels |
| Color Channels | 1 (grayscale) |
| Categories | 10 fashion product types |

### Product Categories

| ID | Category |
|----|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Implementation Details

### Development Environment

**Primary Language:** Python 3

**Core Libraries:**
- TensorFlow/Keras - Deep learning framework
- Scikit-learn - Traditional ML toolkit
- NumPy - Numerical computations
- Matplotlib/Seaborn - Visualization
- Pandas - Data management

### Model 1: Convolutional Neural Network

The CNN architecture employs hierarchical feature extraction through successive convolutional operations, enabling the network to learn increasingly abstract visual patterns from raw pixel data.

**Network Structure:**

```python
model = keras.Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 10
- Total Parameters: 93,322

### Model 2: Random Forest Classifier

The Random Forest operates as a ensemble of decision trees, each trained on bootstrapped data with randomized feature selection to promote diversity among trees.

**Parameter Settings:**
- Number of Trees: 100
- Split Criterion: Entropy
- Maximum Depth: 100
- Bootstrapping: Enabled
- Parallel Execution: Multi-core CPU
- Random State: Fixed for reproducibility

---

## Data Preparation Pipeline

### Common Preprocessing Steps

- **Normalization:** Pixel values scaled from [0, 255] to [0, 1] range
- **Dataset Splitting:** 48,000 training, 12,000 validation, 10,000 test

### Model-Specific Adaptations

**CNN Input Preparation:**
- Preserved spatial structure: reshaped to (28, 28, 1)
- Maintained 2D relationships between neighboring pixels

**RF Input Preparation:**
- Flattened to 784-dimensional vectors
- Lost spatial context (pixels treated as independent features)

---

## Evaluation Framework

### Performance Metrics

The following quantitative measures were employed to assess model effectiveness:

- **Accuracy:** Overall correct prediction rate
- **Precision:** Quality of positive predictions
- **Recall:** Completeness of positive identification
- **F1-Score:** Harmonic balance between precision and recall
- **Confusion Matrices:** Detailed error pattern visualization
- **Training Duration:** Computational resource assessment

### Result Export

All evaluation outputs are automatically saved as:
- PNG images (confusion matrices)
- CSV files (classification reports with per-class metrics)

---

## Results Analysis

### Comparative Performance Metrics

| Metric | CNN Results | RF Results |
|--------|-------------|------------|
| Training Accuracy | 95.81% | 100.00% |
| Validation Accuracy | 91.25% | 88.52% |
| Test Accuracy | 91.03% | 87.49% |
| Training Duration | 98.6 seconds | 11.1 seconds |
| Overfitting Evidence | Minimal | Severe |
| Generalization Quality | Strong | Weak |

**Performance Gap:** The CNN outperforms the RF by 3.54 percentage points on test data.

### Class-Level Performance Patterns

**Well-Classified Categories (Both Models):**
- Trouser, Bag, Sandal, Sneaker, Ankle boot

**Problematic Categories (Both Models):**
1. **Shirt** (most challenging)
2. **Pullover**
3. **Coat**
4. T-shirt/top

**Root Cause Analysis:** The low-resolution (28×28) grayscale format provides insufficient detail to distinguish between upper-body garments with similar silhouettes. Texture, color, and fine structural details—critical for differentiating between a shirt, pullover, and coat—are largely absent.

---

## Model Diagnostics

### Overfitting Investigation

**Random Forest Overfitting Indicators:**
- Perfect training accuracy (100%)
- Substantial test accuracy drop (12.51 percentage points)
- Aggressive decision boundary formation
- Spatial structure loss through flattening

**CNN Behavior:**
- Moderate train-test gap (4.78 percentage points)
- Stable validation performance
- Robust spatial feature learning
- Better generalization through hierarchical representations

### Error Pattern Analysis

The confusion matrices reveal systematic misclassifications:
- **Shirt ↔ T-shirt/top:** Silhouette similarity
- **Pullover ↔ Coat:** Overlapping garment structure
- **Coat ↔ Shirt:** Similar shoulder and sleeve configurations

---

## Practical Recommendations

### Production Deployment Choice

**Recommended: CNN for Production Systems**

**Rationale:**
- Superior classification accuracy (+3.54%)
- Better handling of ambiguous categories
- Robust generalization to unseen data
- Hierarchical feature learning capability
- Spatial relationship preservation

**Appropriate RF Use Cases:**
- Rapid prototyping and baseline establishment
- Resource-constrained environments (CPU-only, limited memory)
- Time-critical applications requiring sub-minute training
- Scenarios where spatial patterns are less critical

---

## Future Enhancement Opportunities

### Potential Improvements

**CNN Enhancements:**
- Hyperparameter optimization (grid search, Bayesian)
- Data augmentation techniques (rotation, zoom, shifting)
- Deeper architectures (ResNet, DenseNet variants)
- Transfer learning from pre-trained networks

**General Improvements:**
- GPU acceleration for faster training
- Deployment as web service (Flask, TensorFlow Serving)
- Application to color fashion datasets
- Real-time inference optimization

---

## Execution Guide

### Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification
```

### Environment Configuration

```bash
# Create virtual environment
python -m venv fashion_env

# Activate environment
# Windows:
fashion_env\Scripts\activate
# Linux/Mac:
source fashion_env/bin/activate
```

### Dependency Installation

Create `requirements.txt` with:

```
tensorflow>=2.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

Install packages:

```bash
pip install -r requirements.txt
```

### Running the Experiment

```bash
python fashion_classifier.py
```

**Execution Outputs:**
- Automatic dataset download
- Model training with progress feedback
- Metric calculations and reporting
- Confusion matrix image generation (PNG)
- Classification report export (CSV)

---

## Project Structure

```
fashion-mnist-classification/
├── fashion_classifier.py          # Main execution script
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── output/                        # Generated results directory
│   ├── CNN_train_confusion_matrix.png
│   ├── CNN_test_confusion_matrix.png
│   ├── RF_train_confusion_matrix.png
│   ├── RF_test_confusion_matrix.png
│   ├── CNN_train_classification_report.csv
│   ├── CNN_test_classification_report.csv
│   ├── RF_train_classification_report.csv
│   └── RF_test_classification_report.csv
└── ...
```

---

## Conclusions

This investigation demonstrates that **CNN-based approaches significantly outperform Random Forest ensembles** for fashion image classification, achieving superior accuracy (91.03% vs 87.49%) and more reliable generalization. The CNN's ability to preserve and exploit spatial relationships through hierarchical feature learning provides a decisive advantage over pixel-independent methods.

While the Random Forest offers substantially faster training (approximately 9× speed advantage), the CNN's predictive reliability makes it the **preferred choice for production environments** where classification accuracy directly impacts user experience and business outcomes.

---

## Academic References

1. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.

2. TensorFlow (2024). Basic classification: Classify images of clothing. *TensorFlow Tutorials*. [Online].

3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.

5. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*.

---

## Licensing

This project is released for educational and research purposes. Users are free to adapt and build upon the code for academic or personal use.

---

*For questions, suggestions, or collaboration opportunities, please submit an issue through the repository.*# Fashion-MNIST Classification: CNN vs Random Forest

## Overview

This project benchmarks a deep learning approach against a traditional machine learning method for automated fashion image classification using the **Fashion-MNIST** dataset.

The study evaluates and compares:

- **Convolutional Neural Network (CNN)** – implemented with TensorFlow/Keras  
- **Random Forest (RF)** – implemented with Scikit-learn  

The goal is to analyze both models in terms of:

- Classification accuracy and generalization
- Precision, recall, and F1-score
- Overfitting behavior
- Computational performance (training time)
- Category‑level prediction reliability

The results demonstrate that the CNN significantly outperforms the Random Forest, achieving **91.03%** test accuracy vs. **87.49%** for the RF, while the RF trains ~9× faster (11.1s vs. 98.6s). The CNN’s superior ability to exploit spatial relationships makes it the recommended choice for production systems, though the Random Forest remains a viable option for rapid prototyping.

---

## Dataset

The project uses the **Fashion‑MNIST** dataset introduced by Xiao et al. (2017).

| Property              | Value                     |
|-----------------------|---------------------------|
| Total images          | 70,000 grayscale          |
| Training images       | 60,000                    |
| Test images           | 10,000                    |
| Image size            | 28×28 pixels              |
| Classes               | 10 fashion categories     |
| Split (this project)  | 48,000 train / 12,000 validation / 10,000 test |

### Categories

| Label | Category      |
|-------|---------------|
| 0     | T‑shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

---

## Technologies Used

- **Language:** Python 3
- **Libraries & Frameworks:**
  - TensorFlow / Keras (CNN)
  - Scikit‑learn (Random Forest)
  - NumPy, Pandas (data handling)
  - Matplotlib, Seaborn (visualisation)
  - CSV (exporting results)

All code is contained in a single Python script (`fashion_classifier.py`) for reproducibility.

---

## CNN Architecture

The CNN was designed to extract spatial hierarchies of features from the 28×28 grayscale images.

```python
model = keras.Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

## Random Forest Configuration

The Random Forest was implemented as a traditional ensemble baseline.

**Parameters:**

- 100 decision trees
- Entropy splitting criterion
- Maximum depth = 100
- Bootstrapped samples with feature randomness at each split
- Parallel processing across all CPU cores
- Fixed random state for reproducibility

Images were **flattened** into 784‑dimensional feature vectors to meet the tabular input requirements of Scikit‑learn.

---

## Data Preprocessing

### For both models:
- Pixel values normalised from [0, 255] to [0, 1] (numerical stability)

### CNN‑specific:
- Reshaped to (28, 28, 1) tensors to preserve spatial dependencies

### RF‑specific:
- Flattened to 784‑dimensional vectors (spatial relationships lost)

---

## Evaluation Metrics

The following metrics were computed for both training and test sets:

- **Accuracy** – overall correctness
- **Precision** – proportion of predicted positives that are correct
- **Recall** – proportion of actual positives correctly identified
- **F1‑Score** – harmonic mean of precision and recall
- **Confusion Matrices** – to visualise misclassification patterns
- **Training Time** – to compare computational efficiency

All results are automatically exported as PNG images and CSV files.

---

## Results Summary

| Metric                  | CNN         | Random Forest |
|-------------------------|-------------|---------------|
| **Training Accuracy**   | 95.81%      | 100.00%       |
| **Validation Accuracy** | 91.25%      | 88.52%        |
| **Test Accuracy**       | **91.03%**  | 87.49%        |
| **Training Time**       | 98.6 sec    | **11.1 sec**  |
| **Generalisation Gap**  | Small       | Large         |
| **Overfitting**         | Low         | High          |

> **CNN Test Accuracy Improvement:** +3.54 percentage points over RF.

---

## Key Findings

### CNN Strengths
- Preserves spatial relationships via convolutional filters
- Learns hierarchical features (edges, textures, shapes) automatically
- More stable generalisation – small gap between train and test performance
- Handles visually similar categories better (e.g., Shirt vs. Pullover)

### RF Strengths
- Extremely fast training (~9× faster than CNN)
- Simpler implementation and lower computational cost
- Works well on distinctive classes (Trouser, Bag, Sneaker, etc.)

### Common Difficult Categories

Both models struggled with:

- **Shirt** (lowest recall for both)
- **Pullover**
- **Coat**
- (also T‑shirt/top to a lesser degree)

These upper‑body garments share similar silhouettes and lack distinctive texture/colour in 28×28 grayscale, making them hard to separate.

---

## Overfitting Analysis

The Random Forest exhibited **clear overfitting**:

- Training accuracy: 100% (perfect memorisation)
- Test accuracy: 87.49% (significant drop)

This occurs because decision trees can partition the feature space aggressively, and the flattening step discards spatial context, forcing the model to rely on independent pixel values.

In contrast, the CNN maintains a smaller train‑test gap, demonstrating robust feature extraction and better generalisation.

---

## Conclusion

The CNN significantly outperforms the Random Forest on the Fashion‑MNIST benchmark, achieving higher test accuracy, better F1‑scores on challenging categories, and more consistent generalisation. Although the RF is much faster to train, the CNN's accuracy advantage makes it the **preferred choice for production‑grade fashion classification systems**.

The RF remains useful for:

- Rapid prototyping or baseline comparisons
- Environments with severe computational or time constraints
- Tasks where spatial structure is less critical

---

## Future Improvements

Potential enhancements include:

- Hyperparameter tuning for both models (grid search / Bayesian optimisation)
- Data augmentation (rotation, shifts, zoom) to boost CNN generalisation
- Deeper or residual architectures (e.g., ResNet)
- Transfer learning on larger, colour fashion datasets (e.g., DeepFashion)
- GPU acceleration for faster training
- Deployment as a lightweight web service (e.g., TensorFlow Serving, Flask)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification
```

### 2. Set up a Python virtual environment (optional but recommended)

```bash
python -m venv fashion_env
source fashion_env/bin/activate       # Linux/Mac
fashion_env\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** should contain:

```
tensorflow>=2.10.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### 4. Run the script

```bash
python fashion_classifier.py
```

The script will:

- Download the Fashion‑MNIST dataset automatically
- Preprocess the data
- Train both models and report metrics
- Generate confusion matrix images (PNG) and CSV files with precision/recall tables

All output files are saved in the current working directory.

---

## Repository Structure

```
.
├── fashion_classifier.py          # Main script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── output/                        # (auto‑generated)
│   ├── CNN_train_confusion_matrix.png
│   ├── CNN_test_confusion_matrix.png
│   ├── RF_train_confusion_matrix.png
│   ├── RF_test_confusion_matrix.png
│   ├── CNN_train_classification_report.csv
│   ├── CNN_test_classification_report.csv
│   ├── RF_train_classification_report.csv
│   ├── RF_test_classification_report.csv
│   └── ... (confusion matrices in CSV)
└── ...
```

---

## License

This project is for educational and research purposes. Feel free to use and adapt it.

---

## Main References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking
machine learning algorithms. arXiv preprint arXiv:1708.07747. https://arxiv.org/pdf/1708.07747

- Tensorflow (2024). Basic classification: Classify images of clothing. https://www.tensorflow.org/tutor
ials/keras/classification [Accessed: 01.06.2026]

---

*For any questions or suggestions, please open an issue or contact the author.*
