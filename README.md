# EN3150 Assignment 03: CNN for Image Classification

This repository contains the complete code for **EN3150 Assignment 03: Simple convolutional neural network to perform classification**. This project implements a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras, compares its performance against `Standard SGD` and `SGD with Momentum`, and then benchmarks this custom model against two state-of-the-art pre-trained models (`ResNet50` and `DenseNet121`) using PyTorch and transfer learning.


[Image of a convolutional neural network architecture]


## 📝 Project Objective

The primary goal of this assignment is to:
1.  **Build a Custom CNN:** Construct a deep CNN from scratch to classify images from the "RealWaste" dataset.
2.  **Analyze Optimizer Performance:** Systematically compare the performance of the `Adam` optimizer against `Standard SGD` and `SGD with Momentum` to understand their impact on training.
3.  **Implement Transfer Learning:** Fine-tune two state-of-the-art models, `ResNet50` and `DenseNet121`, on the same dataset.
4.  **Compare and Conclude:** Analyze the trade-offs in performance, speed, and complexity between the custom-built model and the pre-trained models.

## 🗂️ Dataset

* **Source:** This project uses the **"RealWaste"** dataset, which contains images of waste classified into 9 categories (e.g., plastic, paper, metal).
* **Data Split:** As required by the assignment, the dataset was split into:
    * **Training:** 70%
    * **Validation:** 15%
    * **Testing:** 15%

## 🛠️ Project Structure

This project is divided into two main parts, implemented in two different frameworks.

### Part 1: Custom CNN (from scratch)
* **Framework:** `TensorFlow` / `Keras`
* **Architecture:** A deep VGG-style sequential model consisting of 5 `Conv2D` + `MaxPooling2D` blocks. The filter sizes are (32, 64, 128, 256, 512). This is followed by a `Dense(512)` layer with L2 regularization and a `Dropout(0.2)` layer for classification.
* **Optimizer Comparison:** The model was trained 3 separate times (20 epochs each) to compare:
    1.  **Adam** (lr=1e-4)
    2.  **Standard SGD** (lr=1e-2)
    3.  **SGD with Momentum** (lr=1e-2, momentum=0.9)

### Part 2: State-of-the-Art Model Comparison
* **Framework:** `PyTorch` / `TorchVision`
* **Technique:** Transfer Learning (Fixed Feature Extractor). The pre-trained convolutional bases were frozen, and only the final classifier layer was unfrozen and trained for 20 epochs.
* **Models:**
    1.  **ResNet50**
    2.  **DenseNet121**

## 📊 Final Results Summary

This table summarizes the final performance of the primary custom model and the pre-trained models on the **15% unseen test set**.

| Model | Framework | Test Accuracy |
| :--- | :--- | :--- |
| **Custom CNN (from scratch)** | Keras | **71.95%** |
| **ResNet50 (Fine-Tuned)** | PyTorch | **81.00%** |
| **DenseNet121 (Fine-Tuned)**| PyTorch | **81.00%** |

### Key Findings
* The pre-trained models (`ResNet50` and `DenseNet121`) **significantly outperformed** the custom-built CNN, achieving an identical top accuracy of **81.00%**. This highlights the power of transfer learning, which leverages features learned from the massive ImageNet dataset.
* The custom-built model, which had to learn features from scratch, achieved a solid accuracy of **71.95%**.
* The ~9% performance gap demonstrates the clear advantage of using pre-trained models for common computer vision tasks, saving both training time and yielding superior results.

## 🚀 How to Run

### 1. Requirements
This project uses two different frameworks. The necessary libraries can be installed via `pip`:

```bash
# For PyTorch (Part 2)
pip install torch torchvision

# For TensorFlow (Part 1)
pip install tensorflow

# For evaluation and utilities
pip install scikit-learn numpy matplotlib seaborn pillow
