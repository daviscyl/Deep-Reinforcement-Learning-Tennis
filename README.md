[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Udacity Deep Reinforcement Learning Nanodegree - Project 3: Collaboration and Competition

### Introduction

For this Udacity project, I used a single DDPG agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) multi-agent collaboration environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

Codes were written using Python3.6.2, run this to install necessary packages:

`pip install -r requirements.txt`

The Unity environment `Tennis.app` only runs on Macs, for Windows users, download the Windows 32-bit version [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip).

### Instructions

- Run all the block cells `Report.ipynb` to initiate the environment and go through the trainings.
- Run `evaluate.py` to watch a trained agent play the game with itself.
- `best_actor.pth` and `best_critic.pth` are the weights of the best model's actor and critic networks.
- `network.py` defines the DDPG agent's neural networks.
- `ddpy_agent.py` defines the DDPG agent class.

### Sources
- [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)
- Udacity Learning Materials
