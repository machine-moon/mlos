
# Machine Learning on Linux Setup Guide

## Introduction
This guide will walk you through the installation process of essential tools and libraries needed for machine learning on a Linux system. This includes installing Python, setting up a virtual environment, and installing popular ML libraries such as Pandas, NumPy, PyTorch, Gym, and CUDA.

## Table of Contents
1. [System Update and Python Installation](#system-update-and-python-installation)
2. [Virtual Environment Setup](#virtual-environment-setup)
3. [Installing Pandas and NumPy](#installing-pandas-and-numpy)
4. [Installing PyTorch](#installing-pytorch)
5. [Installing Gym and Its Features](#installing-gym-and-its-features)
6. [CUDA Installation](#cuda-installation)

## System Update and Python Installation
First, ensure your system is up to date and install Python.

### Step 1: Update System
Open your terminal and run the following commands to update your system:

```sh
sudo apt-get update
sudo apt-get upgrade
```

### Step 2: Install Python
Install Python, pip, and venv:

```sh
sudo apt-get install python3 python3-pip python3-venv
```

Verify the installation:

```sh
python3 --version
pip3 --version
```

## Virtual Environment Setup
Create and activate a virtual environment to manage your project dependencies.

### Step 1: Create a Virtual Environment
Run the following command to create a virtual environment:

```sh
python3 -m venv mlos_env
```

### Step 2: Activate the Virtual Environment
Activate the virtual environment:

```sh
source mlos_env/bin/activate
```

### Step 3: Verify Virtual Environment Activation
Your prompt should change to indicate that the virtual environment is active. You can verify this by running:

```sh
which python
```

## Installing Pandas and NumPy
Install Pandas and NumPy using pip.

### Step 1: Install Pandas
Run the following command to install Pandas:

```sh
pip install pandas
```

### Step 2: Install NumPy
Run the following command to install NumPy:

```sh
pip install numpy
```

### Step 3: Verify the Installation
You can verify the installation by running a Python shell and importing the libraries:

```python
import pandas as pd
import numpy as np
print(pd.__version__)
print(np.__version__)
```

## Installing PyTorch
Follow the instructions from the [PyTorch official website](https://pytorch.org/get-started/locally/) to install PyTorch. Below is an example command for installing PyTorch with CUDA support.

### Step 1: Choose the Correct Installation Command
Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) and select your preferences. For example, for CUDA 11.3, you would use:

```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### Step 2: Verify the Installation
Verify the installation by running a Python shell and importing PyTorch:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

## Installing Gym and Its Features
Install Gym and additional environments using pip.

### Step 1: Install Gym
Run the following command to install Gym:

```sh
pip install gym
```

### Step 2: Install Additional Environments
Install additional Gym environments:

```sh
pip install gym[classic_control]
pip install gym[atari]
pip install gym[box2d]
pip install gym[toy_text]
```

### Step 3: Verify the Installation
Verify the installation by running a Python shell and importing Gym:

```python
import gym
print(gym.__version__)
```

## CUDA Installation
CUDA is essential for leveraging the GPU for machine learning tasks. Below are the steps to install CUDA on your system.

### Step 1: Add the NVIDIA Package Repository
```sh
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
```

### Step 2: Install CUDA Toolkit
Update the package lists and install the CUDA toolkit:

```sh
sudo apt-get update
sudo apt-get install -y cuda
```

### Step 3: Set Environment Variables
Add the following lines to your `~/.bashrc` file:

```sh
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Source the `~/.bashrc` file to apply the changes:

```sh
source ~/.bashrc
```

### Step 4: Verify the Installation
Verify the CUDA installation by running:

```sh
nvcc --version
```

### Step 5: Install cuDNN
Download cuDNN from the NVIDIA website and install it. Move the downloaded files to the appropriate directories:

```sh
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Step 6: Verify cuDNN Installation
Verify the cuDNN installation:

```sh
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## Conclusion
You have now set up your machine learning environment on Linux, including Python, virtual environments, Pandas, NumPy, PyTorch, Gym, and CUDA. You are ready to start developing and training machine learning models!

