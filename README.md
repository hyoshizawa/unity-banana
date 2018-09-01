# DQN for Unity banana collector

### Introduction
This is Deep Q-Network agent implementation for Unity banana collector environment. The goal of the agent is to navigate in a large, square space.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. In other words, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.


### Environment
action, reward
The task is episodic.

#### states
The state space has 37 dimensions and contains the agentvelocity, along with ray-based perception of objects around agent's forward direction.

#### actions
Four discrete actions are available, correspoinding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

#### rewards
- **`+1`** - collecting yellow banana
- **`-1`** - collecting blue banana

