# EN3150 Assignment 03: CNN for Image Classification

[cite_start]This repository contains the code and report for the **EN3150 Assignment 03: Simple convolutional neural network to perform classification**[cite: 1]. [cite_start]The project involves building, training, and evaluating a custom Convolutional Neural Network (CNN) and comparing its performance against state-of-the-art pre-trained models using transfer learning[cite: 4, 55].



[Image of a convolutional neural network architecture]


## 📝 Project Objective

The primary goal of this assignment is to:
1.  [cite_start]**Build a Custom CNN:** Construct a simple CNN from scratch to classify images from a chosen dataset[cite: 4, 29].
2.  [cite_start]**Analyze Optimizer Performance:** Compare the performance of the `Adam` optimizer against `Standard SGD` and `SGD with Momentum`[cite: 45].
3.  [cite_start]**Implement Transfer Learning:** Fine-tune two state-of-the-art pre-trained models (e.g., `ResNet50` and `DenseNet121`) on the same dataset[cite: 57, 58].
4.  [cite_start]**Compare and Conclude:** Analyze the trade-offs, advantages, and limitations of the custom-built model versus the pre-trained models[cite: 63, 65].

## 🗂️ Dataset

* [cite_start]**Source:** As per the assignment, the `CIFAR-10` dataset was not used[cite: 7]. This project uses the **"RealWaste"** dataset, which features images of waste classified into different categories (e.g., plastic, paper, metal).
* **Data Split:** The dataset was split according to the assignment's requirements:
    * [cite_start]**Training:** 70% [cite: 9]
    * [cite_start]**Validation:** 15% [cite: 9]
    * [cite_start]**Testing:** 15% [cite: 9]

## 🛠️ Project Structure

This project is divided into two main parts, as outlined in the assignment notebook:

### Part 1: Custom CNN (from scratch)
* **Framework:** `TensorFlow` / `Keras`
* [cite_start]**Architecture:** A VGG-style custom sequential model consisting of 5 `Conv2D` + `MaxPooling2D` blocks, followed by a `Dense` classifier with `Dropout` [cite: 30-37].
* **Optimizer Comparison:** The model was trained 3 separate times (20 epochs each) to compare:
    1.  **Adam** (lr=1e-4)
    2.  **Standard SGD** (lr=1e-2)
    3.  **SGD with Momentum** (lr=1e-2, momentum=0.9)

### Part 2: State-of-the-Art Model Comparison
* **Framework:** `PyTorch` / `TorchVision`
* [cite_start]**Technique:** Transfer Learning (Fixed Feature Extractor)[cite: 56]. The pre-trained convolutional bases were frozen, and only the final classifier layer was trained.
* **Models:**
    1.  **ResNet50**
    2.  **DenseNet121**

## 📊 Final Results Summary

This table summarizes the final performance of all trained models on the **unseen test set**.

| Model | Framework | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **Custom CNN (Adam)** | Keras | **[Your Acc %]** | **[Your Loss]** |
| Custom CNN (SGD) | Keras | [Your Acc %] | [Your Loss] |
| Custom CNN (Momentum) | Keras | [Your Acc %] | [Your Loss] |
| **ResNet50 (Fine-Tuned)** | PyTorch | **[Your Acc %]** | **[Your Loss]** |
| **DenseNet121 (Fine-Tuned)**| PyTorch | **[Your Acc %]** | **[Your Loss]** |

### Key Findings
* **(TODO: Write your finding)**
* **Example:** As expected, the pre-trained `DenseNet121` and `ResNet50` models significantly outperformed the custom-built CNN. This demonstrates the power of transfer learning, which leverages features learned from the massive ImageNet dataset.
* [cite_start]For the custom models, `Adam` and `SGD with Momentum` provided much faster convergence and better final accuracy than standard `SGD`, highlighting the impact of adaptive learning rates and momentum[cite: 49].

## 🚀 How to Run

This project is contained within a single notebook (e.g., `EN3150_A03.ipynb`).

### 1. Requirements
The models were built using standard data science libraries. You can install them via pip:

```bash
pip install -r requirements.txt