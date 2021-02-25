import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import (Actor, Critic)
from noise import OUNoise
from replaybuffer import ReplayBuffer


class DDPG_Agent():
    """Interacts with and learns from the environment."""
#self.state_size, self.action_size, self.seed, hidden_layers_actor, hidden_layers_critic, self.buffer_size, learning_rate_actor, learning_rate_critic
    def __init__(self, state_size, action_size, num_agents, seed, device,
                 buffer_size=int(1e5), batch_size=128, num_batches = 5, update_every=10,
                 gamma=0.99, tau=8e-3,
                 learning_rate_actor=1e-3, learning_rate_critic=1e-3, weight_decay=0.0001,                
                 hidden_layers_actor=[32,32], hidden_layers_critic=[32, 32, 32],
                 add_noise=True, start_eps=5.0, end_eps=0.0, end_eps_episode=500,
                 agent_id=-1):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            seed (int): random seed
            hidden_layers (list of int ; optional): number of each layer nodes
            buffer_size (int ; optional): replay buffer size
            batch_size (int; optional): minibatch size
            gamma (float; optional): discount factor
            tau (float; optional): for soft update of target parameters
            learning_rate_X (float; optional): learning rate for X=actor or critic
        """
        print('In DPPG_AGENT: seed = ', seed)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.device = device
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_batches = num_batches
        
        self.gamma = gamma
        self.tau = tau
        
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.weight_decay_critic = weight_decay
        
        self.add_noise = add_noise
        self.start_eps = start_eps
        self.eps = start_eps
        self.end_eps = end_eps
        self.eps_decay = 1/(end_eps_episode*num_batches)  # set decay rate based on epsilon end target
        self.timestep = 0
        
        self.agent_id = agent_id
     
        ### SET UP THE ACTOR NETWORK ###
        # Assign model parameters and assign device
        model_params_actor  = [state_size, action_size, seed, hidden_layers_actor]
        
        # Create the Actor Network (w/ Target Network)
        self.actor_local = Actor(*model_params_actor).to(self.device)
        self.actor_target = Actor(*model_params_actor).to(self.device)
        #print('actor_local network is: ', print(self.actor_local))
        
        # Set up optimizer for the Actor network
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)       
        
        ### SET UP THE CRITIC NETWORK ###
        model_params_critic = [state_size, action_size, seed, hidden_layers_critic]

        # Create the Critic Network (w/ Target Network)
        self.critic_local = Critic(*model_params_critic).to(self.device)
        self.critic_target = Critic(*model_params_critic).to(self.device)
        
        # Set up optimizer for the Critic Network
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)

        # Noise process
        self.noise = OUNoise(action_size, self.seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)

    def step(self, states, actions, rewards, next_states, dones, agent_number):
        # Increment timestep by 1
        self.timestep += 1
        
        # Save experience in replay memory
        self.memory.add(states, actions, rewards, next_states, dones)
        
         # If there are enough samples and a model update is to be made at this time step
        if len(self.memory) > self.batch_size and self.timestep%self.update_every == 0:
            # For each batch
            for i in range(self.num_batches):
                # Sample experiences from memory
                experiences = self.memory.sample()
        
                # Learn from the experience
                self.learn(experiences, self.gamma, agent_number)

    def act(self, state, scale_noise=True):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        
        # Go to evaluation mode and get Q values for current state
        self.actor_local.eval()
        with torch.no_grad():
            # Get action for the agent and concatenate them
            action = [self.actor_local(state[0]).cpu().data.numpy()]
            
        # get back to train mode
        self.actor_local.train()
        
        # Add noise to the action probabilities
        # Note, we want the magnitude of noise to decrease as the agent keeps learning
        action += int(scale_noise)*(self.eps)*self.noise.sample()
        
        return np.clip(action, -1.0, 1.0)
    
    def reset(self):
        """
        Reset the noise, and all neural network parameters for the current agent
        """
        self.noise.reset()
        self.eps = self.start_eps
        self.timestep = 0
        self.critic_local.reset_parameters()
        self.actor_local.reset_parameters()
        self.critic_target.reset_parameters()
        self.actor_target.reset_parameters()
        
        # ReSet up optimizer for the Actor network
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        
        # Set up optimizer for the Critic Network
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)
        
        # Clear the experience buffer
        self.memory.clear_buffer()
        
    def reset_noise(self):
        """
        Reset the noise only
        """
        self.noise.reset()
   
    def learn(self, experiences, gamma, agent_number):
        ####     DRAW FROM MEMORY AND PREPARE SARS DATA        ####
        # From the experiences buffer, separate out S_t, A_t, R_t, S_t+1, done data
        states, actions, rewards, next_states, dones = experiences
        
        # NOTE: actions has dimension of batch_size x concatenated action for all agents
      
        # get the next action for the current agent for the entire batch
        actions_next = self.actor_target(next_states)
    
        # Construct next action vector for the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        
        ####    UPDATE CRITIC   ####
        # Get predicted next-state actions and Q values from target models
        # Get the next targets
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        # Define the loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip gradient @1
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
   

        # --------------UPDATE ACTOR -----------------------#
        # Compute actor loss
        actions_pred = self.actor_local(states)

        # Construct action prediction vector relative to each agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        
        # Calculate the loss. Note the negative sign since we use steepest ascent
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks using the local and target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, self.end_eps)
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        X_target = tau*X_local + (1 - tau)*X_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class MultiAgent():
    """Interaction between multiple agents in common environment"""
    def __init__(self, 
                 state_size, 
                 action_size, 
                 num_agents, 
                 seed, 
                 buffer_size=int(1e5),
                 batch_size = 128,
                 num_batches = 5,
                 update_every = 10,
                 gamma = 0.99,
                 tau = 1e-3,
                 learning_rate_actor=1e-3, 
                 learning_rate_critic=1e-3,
                 hidden_layers_actor=[32,32],
                 hidden_layers_critic=[32, 32, 32],
                 weight_decay=0.0001,
                 add_noise=True,
                 start_eps = 5.0,
                 end_eps = 0.0,
                 end_eps_episode=500):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_batches = num_batches
        
        # detect GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        
        # Initialize an actor and critic for each agent
        self.agents = []
        for x in range(self.num_agents):
            Agent = DDPG_Agent(state_size, action_size, num_agents, seed, self.device,
                 buffer_size, batch_size, num_batches, update_every,
                 gamma, tau,
                 learning_rate_actor, learning_rate_critic, weight_decay,                
                 hidden_layers_actor, hidden_layers_critic,
                 add_noise, start_eps, end_eps, end_eps_episode, x)
            
            self.agents.append(Agent)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences in replay memory and learn for each agent"""
        for current_agent_number, current_agent in enumerate(self.agents):
            current_agent.step(states, actions, rewards[current_agent_number], next_states, dones, current_agent_number)

    def act(self, states):
        """Agents perform actions according to their policy."""
        # The 1D vector containing the state information for each agent is used to predict each agents action        
        # For each agent and its state, predict the actions as per the policy network and concatenate them
        actions_list = []
        for current_agent_number, current_agent in enumerate(self.agents):
            agents_action = current_agent.act(states, current_agent.add_noise)
            actions_list.append( agents_action )
        action = np.concatenate(actions_list, axis=0)
 
        return action.flatten()
    
    def reset(self):        
        for current_agent in self.agents:
            current_agent.reset()
            
    def reset_noise(self):        
        for current_agent in self.agents:
            current_agent.reset_noise()

    def save_model(self, iternum):
        """Save learnable model's parameters of each agent."""
        for current_agent_number, current_agent in enumerate(self.agents):
            torch.save(current_agent.actor_local.state_dict(), 'agent_actor_checkpoint_{}_{}.pth'.format(current_agent_number + 1, iternum+1))
            torch.save(current_agent.critic_local.state_dict(), 'agent_critic_checkpoint_{}_{}.pth'.format(current_agent_number + 1, iternum+1))
            
    def load_model(self, iternum):
        """Load learnable model's parameters of each agent."""
        for current_agent_number, current_agent in enumerate(self.agents):
            fname_actor = 'agent_actor_checkpoint_{}_{}.pth'.format(current_agent_number+1,iternum+1)
            fname_critic = 'agent_critic_checkpoint_{}_{}.pth'.format(current_agent_number+1,iternum+1)

            current_agent.actor_local.load_state_dict(torch.load(fname_actor))
            current_agent.critic_local.load_state_dict(torch.load(fname_critic))
