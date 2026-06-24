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
