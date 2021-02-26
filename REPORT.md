## Report

The objective of the project was to use self-play in order to teach two RL agents how to play tennis.

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

when the average (over 100 episodes) of those scores over 100 consecutive episodes is at least +0.5, the environment is considered as solved.

### Learning Algorithm

I chose Multi-agent Deep Deterministic Policy Gradient (MADDPG)[https://arxiv.org/abs/1706.02275v4] as the algorithm to "solve" this problem.

It is a simple extension of Deep Deterministic Policy Gradient which is an actor-critic method. 

Each actor acts based on a policy that depends only on its local state space observation whereas each critic acts based on the joint state space observation and actions of each agent.


### Function approximation with Neural Networks.

DDPG is an actor-critic algorithm for continuous actions. 

#### Actor

The Actor is the policy function approximator neural network which takes the current state as input and outputs the action an agent should take. The Actor neural network architecture can be simply described as follows

[State] -> [Hidden layer/384 neurons] -> [Hidden layer/256 neurons] -> Action [output/4-dimensional]

Since the actions are constrained to be in the range [-1,1], the tanh activation function was employed to generate the output. A rectified-linear activation function was used for all other hidden layers

As done for DQN, a local and a target network are used for the Actor. 

#### Critic

The Critic is the action-value function approximator neural network that takes the current state and selected action as input and outputs the action value for the given (state,action) pair. The Critic neural network architecture is shown next

[State,Action] -> [Hidden layer/384 neurons] -> [Hidden layer/256 neurons] -> Action-value function Q(s,a) [output/1-dimension]

Since Q(s,a) is a continuous number, a recified linear activation function was used for all hidden and output layers

A local and a target network were used for the Critic. 

#### Neural Network hyperparameters for both Actor and Critic
- Optimizer: Adam
- Learning rate for Actor: 0.0001
- Learning rate for Critic: 0.0005
- Update target network every : 20 steps

### Replay buffer and learning

- Buffer size: 100,000
- batch size for drawn experience = 128
- number of batches used for training = 10

#### Implementation
The algorithm implementation I used is based on the version provided by Udacity in the Deep Reinforcement Learning Nano-degree program github repository [https://github.com/udacity/deep-reinforcement-learning]


### Exprimental setup

Since neural network optimization and expoloration of the environment are stochastic processes, the experiment should be repeated several times. Accordingly, the following setup was used

- Maximum number of episodes per experiment: 2000
- Maximum number of steps per episode : 1000
- Number of experiments: 2

Ideally I would have used 10 independent experiments to evaluate the performance of the trained agent. However, due to gpu constraints, I performed two independent experiments.
Note that for each independent experiment, the neural network weights were initialized and the experience replay buffer were emptied.


### Criteria for success

When the running average of the per agent per episode reward over the last 100 episodes (if experiment lasts for over 100 episodes) consecutive episodes exceeds 30, the environment is deemed to be solved.

### Results
The number of episodes and total time required to successfully complete each round of the experiment was noted in order to quantify the success.

 The following average metrics were calculated when the experiments successfully completed.

1. Number of episodes required for completion - 65.500000 +/- 39.500000
2. Time (in seconds) required for completion - 8071.121342 +/- 4836.099544

Note that these numbers are not very reliable as they are based on only two experiments.

The performance (as quantified by the score) of the agent for each of these 2 experiments is displayed in the following ![figure](https://github.com/janamejaya/DRLND_Continuous_Control/blob/master/result_score.jpg)

### Future improvements

1. Improving the sampling: Although multiple independent agents did provide more diversity of experiences, perhaps multiple trajectories sampled asynchronously might help
2. Improving the algorithm: Perhaps incorporating double Q-learning, decoupling updates of the actor and critic networks by using different update frequencies, and smoothing the policy over states could help. It would be helpful to understand which aspect of different algorithms help improve the agents performance by getting benchmark results for the Reacher environment.
