# Deep Q-Network: Banana Problem Report

This report describes the implemenation of a deep Q-network (DQN) to solve the banana problem. 

## Learning Algorithm

The banana environment is a Unity environment. 
- State space:
- Action space:
- Rewards:
  
The environment is considered solved when the agent achieves an average reward of at least +13 over the past 100 episodes.

## Results

This implementation solved the problem by achieving an average reward of at least +13 over 100 episodes in less than 600 total episodes. The full results are shown below. The saved weights are available [here](./dqn/checkpoint.pth). 

![Results](./results/score.png)

## Future Work

The agent's performance could be improved by implementing various DQN extensions. The most effective option would be to implement rainbow DQN. 
