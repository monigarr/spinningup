# 🔤 Spinning Up: Modernized PyTorch Edition

**Author**: MoniGarr  
**Branch**: `monigarr-dev`  
**Status**: Stable, updated, and fully operational with modern deep RL libraries.

---

## 🌟 What It Is
A fully modernized and operational fork of [OpenAI's Spinning Up](https://github.com/openai/spinningup), now compatible with current Python (3.8+), PyTorch, Gymnasium, and MuJoCo ecosystems.

This repository restores the value of one of the most respected educational toolkits for deep reinforcement learning (RL) and makes it accessible again to learners, researchers, and developers working in today's environments.

---

## 💡 Why This Matters
OpenAI's original Spinning Up is a brilliant learning resource, but its dependencies are outdated and unusable with modern toolchains *out of the box*  (7/15/2025).

This project revitalizes Spinning Up:
- Updated for modern library compatibility
- Swapped deprecated packages for maintained successors
- Fixed breaking changes in APIs (Gym ➔ Gymnasium)
- Modular and reproducible with Conda + Pip
- Reinforces accessible RL education for diverse researchers

This version is now usable *out of the box* on most modern systems.

---

## 🚀 Quickstart: Install & Run

### 1. Clone the Repository
```bash
git clone https://github.com/monigarr/spinningup.git
cd spinningup
```

### 2. Create & Activate Conda Environment
```bash
conda create -n spinningup-py38 python=3.8
conda activate spinningup-py38
```

### 3. Install Core Dependencies (Conda First)
```bash
conda install -c conda-forge mpi4py openmpi swig -y
```

### 4. Install Python Packages
```bash
pip install "gymnasium[mujoco]"
pip install -e .
```

### 5. Run a PPO Agent LunarLander
```bash
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v3 --exp_name installtest_pytorch --gamma 0.999
```
You should see training logs and results in `./data/installtest_pytorch/`

---

### 6. Run a PPO Agent Walker
This will tell gymnasium to use the modern mujoco engine and load the correct environment.

```bash
python -m spinup.run ppo --env Walker2d-v4 --exp_name walker_pytorch
```
You should see training logs and results in `./data/walker_pytorch/`

Open this project in Visual Studio and you will notice the following new files in your /data/walker_pytorch/walker_pytorch_s0 folder. Each file serves a specific purpose for analyzing, plotting and re-running your trained agent. The following is a breakdown of what you do with each new file.

### 7. Run more Experiments
Review this repository’s discussions for more experiments
https://github.com/monigarr/spinningup/discussions/7 

## config.json
What it is: A text file that saves all the hyperparameters and settings for that specific experiment (e.g., environment name, learning rate, network size).

What you can do with it:

Review Settings: Open it to see the exact configuration you used for a particular run.

Reloading: The test_policy.py script automatically uses this file to know which environment to create when you test your trained agent.

## progress.txt
What it is: This is your primary data file. It's a tab-separated text file that logs all the key performance metrics for each epoch of training, such as AverageEpRet (Average Episode Return), StdEpRet, LossPi, and Entropy.

What you can do with it:

Plot Performance: This file is automatically read by the plot utility to generate your performance graphs.
```bash
python -m spinup.run plot path/to/your/experiment/
```
Custom Analysis: You can open it with a spreadsheet program like Excel or load it into a Python Pandas DataFrame for more detailed, custom analysis.

## pyt_save/model.pt
What it is: This is the most important file. It’s the "brain" of your agent. It contains the saved weights and parameters of your trained PyTorch neural network.

What you can do with it:

Watch Your Agent: Use the test_policy utility to load this file and watch your trained agent perform in the environment.
```bash
python -m spinup.run test_policy path/to/your/experiment/
```

## vars.pkl
What it is: This is a pickle file that saves a snapshot of the training state. In the original TensorFlow version, this was critical for resuming training.

What you can do with it: For your modernized PyTorch workflow, this file is generally not needed. The test_policy script relies on config.json and model.pt to load and run your agent. You can safely focus on the other three files.


---

## 🏠 Architecture Overview

### Core Algorithms
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin-Delayed DDPG)
- DDPG (Deep Deterministic Policy Gradient)
- VPG (Vanilla Policy Gradient)

### Neural Network Design
- Configurable MLPs for actor and critic
- Clean modular structure
- Easy experimentation with policy/value network changes

### Updated Tech Stack
| Component | Modern Version Used |
|----------|----------------------|
| Python   | 3.8                 |
| RL API   | gymnasium           |
| Deep Learning | PyTorch        |
| Physics Engine | mujoco (pip version) |

---

## 📊 Sample Results

Training PPO on Walker2d-v4 after modernization:
```
|      AverageEpRet |          2431.2 |
|          MaxEpRet |          3979.1 |
|          MinEpRet |           432.9 |
|      AverageVVals |          -8.409 |
|           Entropy |           8.524 |
|           EpLen   |           1000  |
|           Time    |         208.71s |
```
To generate your own performance plots:
```bash
python -m spinup.run plot data/test-ppo-lander --savefig ppo_plot.png
```

---

## 🔎 MoniGarr's Motivation
As an Onkwehonwe technologist and AI developer, I’m committed to building bridges between ancient knowledge systems and modern AI research.

This project is part of my broader mission to:
- Make advanced RL research tools usable again for new generations of learners
- Prepare for deeper AI residency-level research work
- Create more accessible and inclusive ML learning pipelines

By modernizing Spinning Up, I’m contributing to an open-source foundation for applied and ethical intelligence research.

---

## 🗺️ Next Steps
- ✅ Extend PPO with attention or recurrent layers
- ✅ Add performance benchmarking for all core algorithms
- ✅ Prepare companion blog series with lessons learned and code walkthroughs
- ✅ Integrate with small research projects (e.g. Kanien’kéha language RL envs)

Follow my journey: [https://github.com/monigarr](https://github.com/monigarr)

---

## 🚑 Contributing & License
This is a research fork for educational purposes. You’re welcome to fork, contribute, or share ideas.

Original source: [OpenAI Spinning Up](https://github.com/openai/spinningup)  
Fork Author: MoniGarr

License: MIT
