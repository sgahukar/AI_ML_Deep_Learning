# 🧠 Deep Learning Case Studies

This project contains two deep learning scenarios implemented using TensorFlow and Keras:

- ✅ **Scenario 1: Image Classification of Handwritten Digits (MNIST Dataset)**
- ✅ **Scenario 2: Binary Classification for Heart Disease Prediction Using ANN**

---

## 📌 Scenario 1: MNIST Handwritten Digit Classification

### 📄 **Problem Statement**
Build a model that recognizes handwritten digits (0–9) from 28x28 grayscale images using the MNIST dataset.

### 🧠 **Model Design**
- Convolutional Neural Network (CNN)
- Layers:
  - `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
  - ReLU activation in hidden layers
  - Softmax activation in the output layer (for multiclass classification)

### ⚙️ **Technologies Used**
- TensorFlow / Keras
- Python
- Matplotlib (for plotting)

### 📊 **Visualizations**
- Sample digit images
- Training vs Validation Accuracy/Loss graphs

### 🚀 **Potential Improvements**
- Data augmentation
- Dropout layers
- Batch normalization
- Use of deeper CNNs like LeNet or ResNet

---

## 📌 Scenario 2: Heart Disease Prediction Using ANN

### 📄 **Problem Statement**
Predict whether a patient is at risk of heart disease based on structured health data (age, cholesterol, blood pressure, etc.).

### 🧠 **Model Design**
- Artificial Neural Network (ANN)
- Layers:
  - `Dense` layers with ReLU activations
  - Final `Dense` layer with sigmoid activation for binary output

### 🎯 **Output Activation Function**
- `Sigmoid` is used to output a probability between 0 and 1, indicating disease risk.

### ⚖️ **Class Imbalance Handling**
- Used `class_weight` during training to handle cases where `has_disease = 1` is underrepresented.

### ⚙️ **Technologies Used**
- TensorFlow / Keras
- Scikit-learn
- Pandas, Numpy
- Matplotlib, Seaborn

### 📊 **Visualizations**
- Training vs Validation Accuracy/Loss graphs
- Confusion Matrix
- ROC Curve and AUC Score

### 🚀 **Potential Improvements**
- Hyperparameter tuning
- Feature selection
- SMOTE oversampling
- Ensemble learning models
 
---

## 📌 How to Run

1. Install required packages:  
   `pip install tensorflow scikit-learn matplotlib pandas`

2. Run each notebook in sequence (recommended in Google Colab or Jupyter).

3. Visual outputs like graphs and model evaluation will be generated automatically.

---

## 👩‍💻 Author
Suhani Gahukar  
Deep Learning | Python | AI Applications  
