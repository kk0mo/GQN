import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch.distributions as dists

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, gmm_k):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim*gmm_k)
        self.fc_lamda = nn.Linear(hidden_dim, gmm_k)
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.gmm_k = gmm_k

    def forward(self, state):
        x = F.softplus(self.fc1(state))
        x = F.softplus(self.fc2(x))
        
        lamda_array = torch.softmax(self.fc_lamda(x), dim=-1)
        lamda_i = torch.multinomial(lamda_array, num_samples=1)
        indices = lamda_i.expand(-1, self.action_dim).unsqueeze(1)

        mu = torch.tanh(self.fc_mu(x)) * self.action_bound
        mu_reshaped = mu.view(-1, self.gmm_k, self.action_dim)
        #print(mu_reshaped)
        
        log_std = torch.full_like(lamda_array, torch.log(torch.tensor(0.1)))
        c1 = torch.tensor(2.8379) # (ln2*pi + 1)
        c2 = torch.tensor(self.action_dim / 2)
        entropy = torch.sum(log_std * lamda_array * c2, dim=1) + c1 * c2

        mu_i = torch.gather(mu_reshaped, 1, indices).squeeze(1)
        covariance_matrix = torch.eye(self.action_dim) * 0.1
        dist = dists.MultivariateNormal(mu_i, covariance_matrix)
        sample = dist.rsample()
        #print(sample)
        
        action = torch.clamp(sample, min = -self.action_bound, max = self.action_bound)
        return action, entropy
    
class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Double_Q_Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc_out1(q1)

        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.fc_out2(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc_out1(q1)
        return q1