# Deep Q-Network: Banana Problem
Deep Q-Network for solving the banana problem. 

## Project Details

The banana environment is a Unity environment that consists of an agent in a room with blue and yellow bananas. The goal is to navigate and collect as many yellow bananas as possible. 

- State space: 37 dimensions including velocity and ray-based perception of objects ahead of the agent.
- Action space: Move forward, backward, left, or right.
- Rewards: +1 (collect yellow banana), -1 (collect blue banana).
  
The environment is considered solved when the agent achieves an average reward of at least +13 over the past 100 episodes.

The code in this project is based heavily off the code from the Udacity Deep Reinforcement Learning Project 1 [base](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) and Deep Q-Network [solution](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

## Getting Started

Follow the instructions at the Udacity [Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning) repository for general insturctions on setting up the environment. Specific instructions for installing and downloading required files for this project are at located in [Project 1](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

## Instructions

Run `dqn/banana.ipynb` to train the agent and visualize the scores over time. The logic for the agent and neural network are in `dgn/dqn_agent.py` and `dqn/qnetwork.py`, respectively. There is also a replay buffer defined at `dqn/replaybuffer.py`. Finally, the `dqn` funciton in `dqn/deepqnetwork.py` manages the training for the agent. The model weights for the successful agent are saved in `dqn/checkpoint.pth`. 


