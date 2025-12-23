# Fashion MNIST Classification using Neural Networks

This project implements a fully connected neural network using **PyTorch** to classify images from the **Fashion-MNIST** dataset.  
The goal is to understand the **end-to-end deep learning pipeline**, from data preprocessing and training to evaluation and real-world image inference.

---

## Project Overview

- Dataset: Fashion-MNIST (10 clothing categories)
- Input: 28×28 grayscale images (flattened to 784 features)
- Output: 10-class classification
- Framework: PyTorch
- Model type: Fully Connected Neural Network (MLP)

---

## Model Architecture

```
Input (784)
 → Linear(784 → 128)
 → BatchNorm + ReLU + Dropout
 → Linear(128 → 64)
 → BatchNorm + ReLU + Dropout
 → Linear(64 → 10)
```

- Loss: Cross Entropy Loss  
- Optimizer: Adam  
- Regularization: Dropout + Batch Normalization  

---

## Training Pipeline

1. Load Fashion-MNIST data
2. Normalize pixel values
3. Train-test split
4. Model training using mini-batch gradient descent
5. Accuracy evaluation on validation and official test set
6. Inference on custom PNG images

---

## Results

- Achieves strong classification accuracy on Fashion-MNIST
- Demonstrates stable convergence with regularization
- Successfully predicts classes on real image inputs

---

## Custom Image Inference

The model supports prediction on images by:
- Converting to grayscale
- Resizing to 28×28
- Normalizing pixel values
- Flattening and passing through the trained model

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Seaborn
- Matplotlib
- KaggleHub

---

## Future Improvements

- Convert to CNN for spatial feature learning
- Add confusion matrix and per-class metrics
- Save and load trained model weights
- Deploy as a simple web app

---

## Author

**Abhi**  
Aspiring Machine Learning & Deep Learning Practitioner  
Focused on understanding models by building systems end-to-end.
