# Fashion-MNIST Classification: CNN vs Random Forest

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
