# Machine Learning on Linux Setup Guide

## Introduction
This guide will walk you through the installation process of essential tools and libraries needed for machine learning on a Linux system. This includes installing Python, setting up a virtual environment, and installing popular ML libraries such as Pandas, NumPy, PyTorch, Gym, and CUDA.

## Table of Contents
1. [Base Folder Explanation](#base-folder-explanation)
2. [Contributing](#contributing)
3. [Setting Up WSL](#setting-up-wsl)
4. [Git](#git)
5. [Virtual Environment Setup](#setting-up-virtual-environments)
6. [Installing PyTorch](#installing-pytorch)
7. [Installing JAX](#installing-jax)
8. [Installing Gym and Its Features](#installing-gym-and-its-features)
9. [CUDA Installation](#cuda-installation)


## Base Folder Explanation

- `cuda/`: Contains CUDA-related installation guides, samples, and tutorials.
- `data/`: Stores raw and processed data.
- `docs/`: Documentation and references, including cheat sheets and research documents.
- `experiments/`: Notebooks, scripts, and results for various experiments.
- `gym/`: Resources related to OpenAI Gym environments.
- `models/`: Model architectures and saved models.
- `nlp/`: NLP model scripts and notebooks.
- `notebooks/`: Jupyter notebooks for experiments and exploratory analysis.
- `reinforcement_learning/`: Reinforcement learning algorithms and resources.
- `scripts/`: Various utility scripts for environment setup, preprocessing, training, and evaluation.
- `transformers/`: Transformer models and related notebooks.
- `vision/`: Computer vision models and notebooks.

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

## CUDA Installation
CUDA is essential for leveraging the GPU for machine learning tasks. Below are the steps to install CUDA on your system.

### Step 1: Download and Install CUDA Toolkit (Base & Driver Installer)
Visit the [CUDA Downloads page](https://developer.nvidia.com/cuda-downloads) and select your operating system, architecture, distribution, and version. Follow the instructions provided for your specific setup.

**Note:** This guide assumes you are working in WSL. The setup process involves installing the base components within WSL, while the GPU driver is installed on the Windows host system.

### Step 2: Install the NVIDIA Driver
For WSL, you need the driver from: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

### Step 3: Install CUDA Toolkit in WSL
Run the following commands in your WSL terminal:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

### Post-installation Actions
These actions must be manually performed after installation before the CUDA Toolkit and Driver can be used.

#### Environment Setup
Add the CUDA path to the PATH variable:

```sh
export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
```

#### Verify the Installation
Verify that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware by compiling and running some of the sample programs, located in [NVIDIA CUDA Samples](https://github.com/nvidia/cuda-samples).

##### Verify the Driver Version
If you installed the driver, verify that the correct version of it is loaded:

```sh
nvidia-smi
nvcc --version
```

### Additional Resources
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [GPU-Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries)
- [WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA Open Source](https://developer.nvidia.com/open-source)
- [CUDA Zone](https://developer.nvidia.com/cuda-zone)
- [NVIDIA NGX](https://docs.nvidia.com/ngx/index.html)

For WSL users, get the Windows driver from [NVIDIA Compute Software Support on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2), then get the toolkit and select WSL2 Ubuntu.



## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
