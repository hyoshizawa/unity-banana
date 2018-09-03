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


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

 2. Setup your Python environment. 
    - Python=3.6
    - torch=0.4.1
    - numpy=1.14.5
    - matplotlib=2.2.3

 3. Run the Report.ipynb and follow instructions.
 