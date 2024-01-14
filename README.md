# Create-Your-Own-Image-Classifier
Udacity X AWS AI with Python Programming Project II

# Image Classification with VGG16

This project implements image classification using the VGG16 model. It includes a training script to train the model and a prediction script to make predictions on new images.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project uses the VGG16 model for image classification. It includes training and prediction scripts written in Python using PyTorch. The training script allows you to train the model on your dataset, while the prediction script allows you to make predictions on new images using a pre-trained model checkpoint.

## Requirements

Before running the scripts, make sure you have the following requirements installed:

- Python 3.x
- PyTorch
- torchvision
- Pillow

## Steps
Install the required libraries using the following command:

```bash
pip install torch torchvision pillow

Usage
Training
To train the model on your dataset, follow these steps:

Organize your dataset into train and valid folders inside a root data directory.

Run the training script:
python train.py data_directory --arch vgg16 --hidden_units 512 --learning_rate 0.001 --epochs 3 --save_dir checkpoints --gpu

Replace data_directory with the path to your root data directory.

Prediction
To make predictions on new images using a pre-trained VGG16 model, follow these steps:

Prepare an image for prediction.

Run the prediction script:
python predict.py input_image checkpoint.pth --category_names cat_to_name.json --gpu

Replace input_image with the path to the image file, checkpoint.pth with the path to your pre-trained model checkpoint, and optionally provide --category_names if you have a JSON file mapping class indices to human-readable class names.

Project Structure
- data/                     # Sample data for testing
- checkpoints/              # Directory to save model checkpoints
- predict.py                # Prediction script
- train.py                  # Training script
- README.md                 # Project documentation
License
This project is licensed under the MIT License - see the LICENSE file for detail
