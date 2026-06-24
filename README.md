# Fashion-MNIST Classification: CNN vs Random Forest

## Overview

This project compares a **Convolutional Neural Network (CNN)** against a **Random Forest (RF)** classifier on the Fashion-MNIST dataset. The CNN achieves superior accuracy (91.03%) while the RF trains 9× faster (11.1s vs 98.6s).

## Dataset

Fashion-MNIST contains 70,000 grayscale images (28×28) across 10 clothing categories:

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

## Models

### CNN Architecture

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

- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy
- **Epochs:** 10
- **Batch size:** 32
- **Parameters:** 93,322

### Random Forest Configuration

- **Trees:** 100
- **Criterion:** Entropy
- **Max depth:** 100
- **Parallel processing:** Multi-core CPU

## Results

| Metric | CNN | Random Forest |
|--------|-----|---------------|
| Training Accuracy | 95.81% | 100.00% |
| Validation Accuracy | 91.25% | 88.52% |
| **Test Accuracy** | **91.03%** | 87.49% |
| Training Time | 98.6s | 11.1s |

### Worst Performing Categories (Test Set)

| Model | #1 | #2 | #3 |
|-------|----|----|-----|
| CNN | Shirt (P:0.784, R:0.675) | Coat (P:0.812, R:0.910) | Pullover (P:0.850, R:0.883) |
| RF | Shirt (P:0.718, R:0.583) | Pullover (P:0.762, R:0.787) | Coat (P:0.758, R:0.806) |

Upper-body garments (Shirt, Pullover, Coat) are consistently misclassified due to similar silhouettes and lack of texture/color in grayscale 28×28 images.

## Key Takeaways

**CNN Strengths:**
- Preserves spatial relationships
- Learns hierarchical features
- Better generalization
- Handles ambiguous categories better

**RF Strengths:**
- 9× faster training
- Lower computational cost
- Simple implementation

## Recommendation

**Use CNN in production** for its superior accuracy and generalization. Use RF for rapid prototyping or resource-constrained environments.

## How to Run

```bash
# Clone repository
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification

# Install dependencies
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn

# Run the script
python fashion_classifier.py
```

## Output Files

The script generates:
- 4 confusion matrix PNGs (CNN/RF × train/test)
- 8 CSV files (classification reports + confusion matrices)

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms.
- TensorFlow (2024). Basic classification: Classify images of clothing.
- Breiman, L. (2001). Random forests. *Machine Learning*.

---

*For questions or suggestions, please open an issue.*
