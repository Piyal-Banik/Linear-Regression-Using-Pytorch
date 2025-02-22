# Linear Regression using PyTorch

## Overview

This repository contains an implementation of a simple linear regression model using PyTorch. The model is trained on randomly generated dummy data and optimized using Stochastic Gradient Descent (SGD).


## Features

- Uses PyTorch's nn.Module to define a linear regression model.

- Generates synthetic data for training.

- Implements a training loop with Mean Squared Error (MSE) loss function.

- Utilizes GPU acceleration if available.


## Installation
Ensure you have Python and the required dependencies installed. You can install the necessary libraries using:

```bash
  pip install torch scikit-learn tqdm
```
    
## Usage/Examples

Clone the repository and navigate to the project directory:

```bash
 git clone https://github.com/Piyal-Banik/Linear-Regression-Using-Pytorch.git
 cd Linear-Regression-Using-Pytorch
```

Run the ipynb notebook:

```bash
 linear_regression.ipynb
```

## Code Breakdown
## Code Breakdown

### 1. Generate Dummy Data
- Creates `X` with 10 features per sample.
- Generates `y` using random weights and bias.
- Splits the dataset into training and test sets.

### 2. Define Model
- Implements a simple linear regression model using `nn.Linear`.

### 3. Training Loop
- Uses the Mean Squared Error (MSE) loss function and Stochastic Gradient Descent (SGD) optimizer.
- Runs for 1000 epochs, printing the loss every 100 epochs.

### 4. Predictions
- Evaluates the model on the test dataset.


## Results
The model learns to approximate the relationship between X and y. Training and test losses are displayed during training.

