import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Policy approximation network."""

    def __init__(self, state_size, action_size, seed, hidden_layers, head_name, head_scale):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): number of nodes for each hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.head_name=head_name
        self.head_scale=head_scale
        
        self.fc1 = nn.Linear( state_size, hidden_layers[0])
        #self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        #self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], action_size)
        
        #self.bn1 = nn.BatchNorm1d(state_size)
        #self.bn2 = nn.BatchNorm1d(hidden_layers[0])
        #self.bn3 = nn.BatchNorm1d(hidden_layers[1])
        
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        #x = F.relu(self.bn1(self.fc1(state)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #return F.tanh(self.bn3(self.fc3(x)))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
        
class Critic(nn.Module):
    """
    input: state(s) and action(a) vectors
    output: the Action-value function approximation i.e a single value
    """

    def __init__(self, state_size, action_size, seed, hidden_layers, head_name, head_scale):
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
        self.head_name=head_name
        self.head_scale=head_scale
        
        input_dim = state_size*2 + action_size*2
        self.fcs1 = nn.Linear(input_dim, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], 1)
        #self.bn1 = nn.BatchNorm1d(input_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_layers[0])
        #self.bn3 = nn.BatchNorm1d(hidden_layers[1])
        
        self.reset_parameters()

        
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            
    def forward(self, state, action):
        #xs = torch.cat((state,action), dim=1)
        #x = F.relu(self.bn1(self.fcs1(xs)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #return F.relu(self.bn3(self.fc3(x)))
    
        xs = torch.cat((state,action), dim=1)
        x = self.bn1(F.relu(self.fcs1(xs)))
        #x = F.relu(self.bn1(self.fcs1(xs)))
        x = self.bn2(F.relu(self.fc2(x)))
        return F.relu(self.fc3(x))   
    
        #xs = torch.cat((state,action), dim=1)
        #x = F.relu(self.fcs1(xs))
        #x = F.relu(self.fc2(x))
        #return F.relu(self.fc3(x))                   
