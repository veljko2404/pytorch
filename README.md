# PyTorch Deep Learning

This repository contains a collection of deep learning experiments implemented using PyTorch.  
The project documents a learning and research journey that progresses from basic PyTorch concepts to more advanced optical character recognition (OCR) architectures.

The repository is organized into several modules, each representing a different stage of experimentation and development.

---

## Repository Structure

### 0_learning_pytorch
Introductory experiments for learning the fundamentals of PyTorch.  
This section contains simple examples that demonstrate how tensors, neural networks, and training loops work inside the framework.

### 1_research_self_distillation
Research-oriented experiments focused on **self-distillation**, a technique where a model improves by learning from its own predictions or internal representations.

### 2_emnist
Experiments using the **EMNIST handwritten character dataset**.  
The code in this folder trains and evaluates models for recognizing handwritten letters and digits.

### 3_ocr_crnn
Implementation of an OCR system based on the **CRNN (Convolutional Recurrent Neural Network)** architecture.  
CNN layers extract visual features from images while recurrent layers model the sequence of characters.

### 4_ocr_cnn_transformer
A more advanced OCR architecture that combines **CNN feature extraction with a Transformer-based sequence model**.  
The CNN processes the image while the Transformer captures long-range dependencies using attention mechanisms.

---

## Technologies

- Python
- PyTorch
- Deep Learning
- Optical Character Recognition (OCR)

---

## Purpose of the Repository

The goal of this repository is to explore and implement modern deep learning techniques while building progressively more advanced OCR models.
