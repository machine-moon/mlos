# Machine Learning on Linux Setup Guide

## Introduction
This guide will walk you through the installation process of essential tools and libraries needed for machine learning on a Linux system. This includes installing Python, setting up a virtual environment, and installing popular ML libraries such as Pandas, NumPy, PyTorch, and Gymnasium.

## Table of Contents
1. [Contributing](#contributing)
2. [Setting Up WSL](#setting-up-wsl)
3. [Git](#git)
4. [Virtual Environment Setup](#setting-up-virtual-environments)
5. [Installing PyTorch](#installing-pytorch)
6. [Installing JAX](#installing-jax)
7. [Installing Gym and Its Features](#installing-gym-and-its-features)


## base Folder Explanation

- `src/gym/`: Resources related to OpenAI Gym environments.
- `src/rl/`: Reinforcement learning algorithms and resources.
- `src/scripts/`: Various utility scripts for environment setup, preprocessing, training, and evaluation.
- `venv/`: Virtual environment directory containing isolated Python environments for the project.

## Contributing

1. **Create a new branch:**

   ```sh
   git checkout -b feature-branch
   ```

2. **Make your changes and commit them:**

   ```sh
   git commit -m "Description of changes"
   ```

3. **Push to the branch:**

   ```sh
   git push origin feature-branch
   ```

4. **Submit a pull request.**

## Setting Up WSL

### Step 1: Set Up WSL 2
Set WSL 2 as the default version:

```sh
wsl --set-default-version 2
```

### Step 2: Install Ubuntu (required)

Open PowerShell as Administrator and run the following command to enable WSL:

```sh
wsl --install -d Ubuntu

```

### Step 3: Update and Upgrade Ubuntu
Open the installed Linux distribution and update it:

```sh
sudo apt-get update
sudo apt-get upgrade
```

### Step 4: Install Essential Tools
Install necessary tools and libraries:

```sh
sudo apt-get install build-essential
sudo apt-get install python3 python3-pip python3-venv
pip install --upgrade pip
```

Verify the installation:

```sh
python3 --version
pip3 --version
```

### Additional Resources
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)


## Git
Create and activate a virtual environment to manage your project dependencies.

### Step 1: Install Git
Install Git if you haven't already:

```sh
sudo apt-get install git
```

### Step 2: Configure SSH Keys
Generate and configure SSH keys for Git:

```sh
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
```

Copy the output and add it to your GitHub account under SSH keys.

### Step 3: Clone the Repository
Clone your project repository:

```sh
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:your-username/mlos.git
cd mlos
```
## Setting Up Virtual Environments
To ensure a consistent and isolated Python environment for this project, it's recommended to use a virtual environment. Follow the steps below to set up a virtual environment and install the necessary dependencies listed in `requirements.txt`.

### Step 1: Create a Virtual Environment
Run the following command to create a virtual environment named `venv`:

```sh
cd ~/workspace/mlos
python3 -m venv venvs/venvName
```

### Step 2: Activate the Virtual Environment

```sh
source venvs/venvName/bin/activate
```

### Step 3: Install Dependencies
Run the following command to install the required packages from `requirements.txt`:

```sh
pip install -r requirements.txt
```

### Step 4: Verify the Installation
You can verify the installation by running a Python shell and importing the libraries listed in `requirements.txt`:

```python
import pandas as pd
import numpy as np
print(pd.__version__)
print(np.__version__)
```
Or by running:

```sh
pip list
```

### Step 5: Deactivate the Virtual Environment
When you are done working in the virtual environment, you can deactivate it using the following command:

```sh
deactivate
```

By following these steps, you can ensure a consistent development environment and avoid potential conflicts with other projects' dependencies.

## Installing PyTorch
Follow the instructions from the [PyTorch official website](https://pytorch.org/get-started/locally/) to install PyTorch. Below is an example command for installing PyTorch with CUDA support.

### Step 1: Choose the Correct Installation Command
Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) and select your preferences. For example, for CUDA 11.3, you would use:

```sh
pip3 install torch torchvision torchaudio
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
```

## Installing JAX
Install JAX and its libraries FLAX, RLAX, and Optax using pip.

### CPU-only (Linux/macOS/Windows)
To install JAX for CPU-only usage, run the following commands:

```sh
pip install --upgrade pip
pip install -U jax
```

### GPU (NVIDIA, CUDA 12)
To install JAX with NVIDIA GPU support, run the following commands:

```sh
pip install --upgrade pip
pip install -U "jax[cuda12]"
```

### TPU (Google Cloud TPU VM)
To install JAX on a Google Cloud TPU VM, run the following commands:

```sh
pip install --upgrade pip
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Installing Machine Learning Libraries  (FLAX, RLAX, Optax)
To install additional JAX libraries such as FLAX, RLAX, and Optax, run the following command:

```sh
pip install flax rlax optax
```

### Verify the Installation
You can verify the installation by running a Python shell and importing the libraries:

```python
import jax
import flax
import rlax
import optax
```

By following these steps, you can ensure that JAX is correctly installed in your virtual environment.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
