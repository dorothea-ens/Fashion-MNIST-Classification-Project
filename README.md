# Fashion-MNIST Classification: CNN vs Random Forest

📄 **[Read the Detailed Project Report](PROJECT_REPORT.md)**

## Overview

This project compares a **Convolutional Neural Network (CNN)** against a **Random Forest (RF)** classifier on the Fashion-MNIST dataset.

The CNN achieves superior test accuracy (91.03% vs. 87.49%), while the Random Forest trains approximately 9× faster (11.1s vs. 98.6s).

The project investigates the trade-offs between predictive accuracy, generalization, computational efficiency, and model complexity in image classification.

## Dataset

Fashion-MNIST contains 70,000 grayscale images (28×28 pixels) across 10 clothing categories:

| Label | Category |
|-------|----------|
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

The dataset is divided into:

- **48,000 training images**
- **12,000 validation images**
- **10,000 test images**

Pixel intensities are normalized to the range [0, 1].

For the CNN, images retain their spatial structure as 28×28×1 tensors. For the Random Forest, images are flattened into 784-dimensional feature vectors.

## Models

### CNN Architecture

The CNN is implemented using TensorFlow/Keras and follows a standard convolutional architecture similar to the [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn).

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

**Training configuration:**

- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy
- **Epochs:** 10
- **Batch size:** 32
- **Parameters:** 93,322

The convolutional layers extract local visual features while preserving spatial relationships between neighboring pixels.

### Random Forest Configuration

The Random Forest classifier is implemented using scikit-learn.

- **Trees:** 100
- **Criterion:** Entropy
- **Max depth:** 100
- **Parallel processing:** Multi-core CPU
- **Input:** Flattened 784-dimensional image vectors

The Random Forest serves as a traditional machine-learning baseline for comparison with the deep-learning approach.

## Results

| Metric | CNN | Random Forest |
|--------|-----|---------------|
| Training Accuracy | 95.81% | 100.00% |
| Validation Accuracy | 91.25% | 88.52% |
| **Test Accuracy** | **91.03%** | 87.49% |
| Training Time | 98.6s | 11.1s |

The CNN achieves a **3.54-percentage-point improvement in test accuracy** over the Random Forest.

Although the Random Forest achieves perfect training accuracy, its substantially lower test accuracy indicates overfitting.

The CNN demonstrates stronger generalization to previously unseen images.

### Worst Performing Categories (Test Set)

| Model | #1 | #2 | #3 |
|-------|----|----|-----|
| CNN | Shirt (P: 0.784, R: 0.675) | Coat (P: 0.812, R: 0.910) | Pullover (P: 0.850, R: 0.883) |
| RF | Shirt (P: 0.718, R: 0.583) | Pullover (P: 0.762, R: 0.787) | Coat (P: 0.758, R: 0.806) |

*P = Precision; R = Recall.*

Upper-body garments (Shirt, Pullover, Coat) are particularly challenging due to similar silhouettes and the absence of detailed color and texture information in low-resolution grayscale images.

Both models exhibit classification difficulties in these categories, although the CNN generally achieves stronger results.

## Key Takeaways

### CNN Strengths

- Preserves spatial relationships between pixels
- Learns hierarchical visual representations
- Achieves higher test accuracy
- Demonstrates better generalization
- Handles visually ambiguous categories more effectively

### Random Forest Strengths

- Approximately 9× faster training
- Lower training computational cost in this experiment
- Straightforward implementation
- Useful baseline for evaluating more complex architectures

### Important Observations

**1. Accuracy versus computational cost**

The CNN achieves better predictive performance but requires substantially more training time.

**2. Overfitting**

The Random Forest's perfect training accuracy does not translate into equally strong test performance, highlighting the importance of evaluating models on unseen data.

**3. Feature representation**

CNNs preserve local spatial information, whereas the Random Forest processes flattened pixel vectors. This contributes to the performance difference observed in the experiment.

**4. Category-specific limitations**

Both models struggle with visually similar garments, demonstrating the limitations of classification using low-resolution grayscale images.

## Recommendation

The CNN is the stronger candidate for applications prioritizing classification accuracy and generalization.

The Random Forest remains useful for rapid prototyping and as a computationally inexpensive baseline.

However, the experiment uses the standardized Fashion-MNIST benchmark rather than real-world retail imagery. Production deployment would require additional testing with representative data, including variations in backgrounds, lighting, resolution, and garment appearance.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/dorothea-ens/Fashion-MNIST-Classification-Project.git
cd Fashion-MNIST-Classification-Project
```

### 2. Install dependencies

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### 3. Run the script

```bash
python fashion_classifier.py
```

## Output Files

The script generates:

- 4 confusion matrix PNGs (CNN/RF × training/test)
- 8 CSV files containing classification reports and confusion matrices

These outputs support the evaluation of overall accuracy, class-level precision and recall, and systematic misclassification patterns.

## Project Documentation

For a more detailed technical discussion, including experimental methodology, model evaluation, overfitting analysis, computational trade-offs, limitations, and academic references, see:

**[Detailed Project Report — PROJECT_REPORT.md](PROJECT_REPORT.md)**

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms.* [arXiv:1708.07747](https://arxiv.org/abs/1708.07747).
- Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324
- TensorFlow. [Basic classification: Classify images of clothing](https://www.tensorflow.org/tutorials/keras/classification).
- TensorFlow. [Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn).
- Scikit-learn. [RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## AI Assistance Disclosure

AI language models were used as assistive tools during the development and documentation process, including conceptual guidance, code optimization support, and preparation of explanatory materials.

The project was reviewed and evaluated by the author, who remains responsible for the implementation, reported findings, and interpretations.

---

*For questions or suggestions, please open an issue.*
