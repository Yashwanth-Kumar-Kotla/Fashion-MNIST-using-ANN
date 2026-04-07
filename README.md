<div align="center">
  
# 👕 Fashion MNIST Classifier using Artificial Neural Networks (ANN)
**High-performance image classification implemented in PyTorch.**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

## 📌 Project Overview
This project implements a robust Artificial Neural Network (ANN) using **PyTorch** to classify clothing images from the Fashion-MNIST dataset. Developed with a focus on modern deep learning practices, the model leverages GPU acceleration (CUDA) and includes advanced regularization techniques to achieve high prediction accuracy while preventing overfitting.

This repository serves as a showcase of end-to-end deep learning workflow: from data loading and preprocessing using custom `Dataset` and `DataLoader` classes, to model building, hyperparameter tuning, and performance evaluation.

---

## 🛠️ Technology Stack
- **Framework:** PyTorch (with CUDA support for fast training)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (data splitting, evaluation)
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebooks

---

## 🧠 Model Architecture & Technical Highlights
The model is a fully connected Deep Neural Network designed with best practices in mind, featuring:

- **Multi-layer Perceptron (MLP)** with feature mapping (784 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 10)
- **Batch Normalization** (`nn.BatchNorm1d`) for accelerated training and internal covariate shift reduction.
- **Dropout Layers** (`p=0.3`) for effective regularization to combat overfitting.
- **Non-linear Activations** (`nn.ReLU`) providing network plasticity.
- **Loss Optimization** via `nn.CrossEntropyLoss` and Adam/SGD optimizers.

```python
# Core Model Definition Highlights
self.model = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 10)
)
```

---

## 🚀 Key Skills Demonstrated
- Implementation of **Custom PyTorch Datasets** and **DataLoaders** for efficient batching.
- Dimensionality manipulation and tensor reshaping.
- Normalization of pixel values to scale $[0, 1]$ converging gradients faster.
- Implementation of the **Training Loop and Evaluation Loop** independently, tracking metrics over epochs.
- Transferring models and tensors dynamically to **GPU (CUDA)** for parallel processing.

---

## 📊 Dataset: Fashion MNIST
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes (e.g., T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

*(Note: The raw CSV datasets are ignored via `.gitignore` to keep the repository lightweight and adhering to Git best practices. Ensure you download `fashion-mnist_train.csv` prior to running the notebooks locally).*

---

## 💻 Usage Instructions
1. Clone the repository: `git clone https://github.com/Yashwanth-Kumar-Kotla/Fashion-MNIST-using-ANN.git`
2. Download the `fashion-mnist_train.csv` dataset and place it in the root directory.
3. Open `Fashion_MNIST_GPU.ipynb` to view the training process and execute the cells step-by-step.

## ✉️ Author
**Yashwanth Kumar Kotla**  
Passionate about software engineering, data science and pushing the boundaries of machine learning.  
[GitHub Profile](https://github.com/Yashwanth-Kumar-Kotla)
