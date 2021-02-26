## Report

The objective of the project was to use self-play in order to teach two RL agents how to play tennis. 

A side note on the environment:
In real tennis, the ball is allowed to bounce off the surface of the court unlike the tennis environment which penalizes such an outcome. Hence, the tennis environment is actually good for teaching two agents how to play Badminton !

### State space

The observation space for each agent consists of 8 variables corresponding to the position and velocity of the ball and racket i.e 8x3 = 24 variables. Each agent receives its own, local observation. So there are a total of 24x2 = 48 variables jointly specifying the environment for the two agents.

### Action space
For each agent, two continuous actions are available and these correspond to movement toward (or away from) the net, and jumping. So there are a total of 4 action variables, 2 per agent.

### Reward

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

### Task description and successful completion
The task is episodic and ends when any agent recieves a reward of -0.01.

After each episode ends, we add up the rewards that each agent received (**without discounting**), to get a score for each agent. This yields 2 (potentially different) scores. 

We then take the maximum of these 2 scores. This yields a single score for each episode.

When the average (over 100 episodes) of those scores over 100 consecutive episodes is at least +0.5, the environment is considered as solved.

### Learning Algorithm

I chose Multi-Agent Deep Deterministic Policy Gradient (MADDPG)[https://arxiv.org/abs/1706.02275v4] as the algorithm to "solve" this problem.

It is a simple extension of Deep Deterministic Policy Gradient which is an actor-critic method. 

Each actor acts based on a policy that depends only on its local state space observation whereas each critic acts based on the joint state space observation and actions of each agent. The algorithm is summarized below

![](https://github.com/janamejaya/DRLND-Collaboration_and_Competition/blob/master/maddpg.png)

### Function approximation with Neural Networks.

MADDPG is an actor-critic algorithm for continuous actions and requires a neural network function approximator for the actor and critic for each agent

#### Actor

The Actor is the policy function approximator neural network which takes the current state as input and outputs the action an agent should take. The Actor neural network architecture can be simply described as follows

[State] -> [Hidden layer/256 neurons] -> [Hidden layer/128 neurons] -> Action [output/2-dimensional]

Since the actions are constrained to be in the range [-1,1], the tanh activation function was employed to generate the output. A rectified-linear activation function was used for all other hidden layers

As done for DQN, a local and a target network are used for the Actor. 

#### Critic

The Critic is the action-value function approximator neural network that takes the current state and selected actions of both agents as input and outputs the action value for the given agent. The Critic neural network architecture is shown next

[State,Action] -> [Hidden layer/256 neurons] -> [Hidden layer/128 neurons] -> Action-value function Q(s,a) [output/1-dimension]

Since Q(s,a) is a continuous number, a recified linear activation function was used for all hidden and output layers

A local and a target network were used for the Critic. 

#### Neural Network hyperparameters for both Actor and Critic
- Optimizer: Adam
- Learning rate for Actor: 0.001
- Learning rate for Critic: 0.001
- Update target network every : 10 steps

### Replay buffer and learning

- Buffer size: 100,000
- batch size for drawn experience = 512
- number of batches used for training = 5

### Noise setting
Noise generated from a stochastic Ornstein–Uhlenbeck process is added to the output of the Actor network and the output clipped to be in the range [-1:1]

#### Implementation
The algorithmic implementation I used is based on the version provided by Udacity in the Deep Reinforcement Learning Nano-degree program github repository [https://github.com/udacity/deep-reinforcement-learning] and modifed to provide a unified interface to the two agents.

### Exprimental setup

An experiment is defined as starting the learning process from the first episode until the successful completion after a finite number of episodes or unsuccessful attempts over a predefined maximum number of episodes.

At the start of an experiment
- the neural network weights were reinitialized
- the optimizer state was reinitialized
- the replay buffer was emptied
- the long running mean noise was reset to zero
- the starting amplitude of noise was reset to the initial maximum value

Since neural network optimization and exploration of the environment are stochastic processes, each experiment involves a different set of trajectories. To get meaningful statistics, the experiment should be repeated several times. Accordingly, the following setup was used

- Maximum number of episodes per experiment: 5000
- Maximum number of steps per episode : unconstrained as we want to agents to sample the terminal state whenever it is attained.
- Number of independent experiments: 2

Ideally I would have used many independent experiments to evaluate the performance of the trained agents. However, due to gpu constraints, I performed 2 independent experiments in the Udacity Workspace and 10 on my local computer.

### Issues with successful completion

Successful completion required a lot of playing around with the neural network architecture (number of neurons in each layer, batch normalization or dropout) for the actor and critic networks, optimization strategy (learning rate), exploration of action (maximum noise), batch size, number of batches, update parameter tau that controls how much the local network should contribute to the update of the target network, etc.

For a large number of combinations, the score would fluctuate such that the 100-episode average score either did not reach even a value of 0.1, or reached a value more than 0.1 but smaller than the target value and dropping off to a much smaller value close to zero (if not zero).

Finally, I observed that using a large value for the maximum amplitude of noise, excluding batch norm or dropout, using a smaller neural network, and a mixing value of tau=0.01 started lead to a larger fraction of experiments that ended successfully.

### Results
The number of episodes and total time required to successfully complete each round of the experiment was noted in order to quantify the success.

The following average metrics were calculated when the 2 experiments successfully completed:

1. Number of experiments that were completed successfully - 2/2
2. Number of episodes required for successful completion - 2169.500000 +/- 2.500000
3. Time (in seconds) required for successful completion - 2401.462885 +/- 128.691792

Note that these numbers are not very reliable as they are based on only two experiments.

The performance (as quantified by the score) of the agent for each of these 2 experiments is displayed in the following ![figure](https://github.com/janamejaya/DRLND-Collaboration_and_Competition/blob/master/result_score.jpg)

For comparison, the average metrics over 10 independent experiments performed on my own computer are:
1. Number of experiments that were completed successfully - 9/10
2. Number of episodes required for successful completion - 2164.000000 +/- 159.979860
3. Time (in seconds) required for successful completion - 1517.755533 +/- 104.106180

The results for the local version of the experiments are displayed below ![figure](https://github.com/janamejaya/DRLND-Collaboration_and_Competition/blob/master/result_score_local_10_runs.jpg)

Since the Udacity workspace and my computer use different GPUs, the time for completion cannot be compared.
However, the average number of episodes required for successful completion are approximately the same.

### Future improvements

Some potential directions to consider in subsequent experiments are 
1. Use a prioritized replay buffer for better sampling of experiences to train the agents
2. One reason for the agent to not perform well was a policy that deteriorated. Perhaps using PPO or similar algorithm for each agent might help.
3. Introducing double Q-learning could also help.
4. Generating experiences from multiple copies of the environment would also help improve the time completion since diverse experiences could be generated faster.

I look forward to trying out these ideas!

