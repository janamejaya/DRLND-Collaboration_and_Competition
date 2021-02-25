import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        inp_dim = 2*state_size
        self.fc1 = nn.Linear(inp_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic2(nn.Module):
    """
    input: state(s) and action(a) vectors
    output: the Action-value function approximation i.e a single value
    """

    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): number of nodes for each hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        input_dim = state_size*2 + action_size*2
        self.inp1 = nn.Linear(state_size, state_size)
        self.inp2 = nn.Linear(state_size, state_size)
        self.inp3 = nn.Linear(action_size, action_size)
        self.inp4 = nn.Linear(action_size, action_size)

        self.fcs1 = nn.Linear(input_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], 1)        
        self.reset_parameters()

        
    def reset_parameters(self):
        self.inp1.weight.data.uniform_(*hidden_init(self.inp1))
        self.inp2.weight.data.uniform_(*hidden_init(self.inp2))
        self.inp3.weight.data.uniform_(*hidden_init(self.inp3))
        self.inp4.weight.data.uniform_(*hidden_init(self.inp4))
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            
    def forward(self, state, action):
        # Split state and action tensors into equal halves
        state1,state2 = torch.chunk(state,2,dim=1)
        action1,action2 = torch.chunk(action,2,dim=1)

        x1 = F.relu(self.inp1(state1))
        x2 = F.relu(self.inp2(state2))
        x3 = F.relu(self.inp3(action1))
        x4 = F.relu(self.inp4(action2))

        xall = torch.cat((x1,x2,x3,x4), dim=1)
        x = F.relu(self.fcs1(xall))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))   


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size*2, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0]+(action_size*2), hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
