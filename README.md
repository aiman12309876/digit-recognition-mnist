# Handwritten Digit Recognition

## Description
A Convolutional Neural Network (CNN) that recognizes handwritten digits from the MNIST dataset.

## Features
- CNN with 3 convolutional layers
- 99%+ accuracy
- Real-time prediction on sample digits

## Dataset
- MNIST
- 70,000 images (28x28 pixels)
- Digits 0-9

## Model Architecture
- Conv2D(32) + MaxPooling
- Conv2D(64) + MaxPooling
- Conv2D(64)
- Flatten + Dense(64) + Dense(10)

## Installation
```bash
pip install tensorflow matplotlib numpy