import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import torch.distributions as dists
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound, gmm_k):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim*gmm_k)
        self.fc_lamda = nn.Linear(hidden_dim, gmm_k)
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.gmm_k = gmm_k

    def forward(self, state):
        x = F.softplus(self.fc1(state))
        #x = F.softplus(self.fc2(x))
        lamda_array = torch.softmax(self.fc_lamda(x), dim=-1)
        #print(lamda_array)
        lamda_i = torch.multinomial(lamda_array, num_samples=1)
        #lamda_i = lamda_i.squeeze(-1)
        #indices = lamda_i.unsqueeze(1).expand(-1, self.action_dim).unsqueeze(1)
        indices = lamda_i.expand(-1, self.action_dim).unsqueeze(1)
        mu = torch.tanh(self.fc_mu(x))
        mu_reshaped = mu.view(-1, self.gmm_k, self.action_dim)
        #print(mu_reshaped.shape)
        mu_i = torch.gather(mu_reshaped, 1, indices).squeeze(1)
        #print(mu_i.shape)
        #scale = torch.full(mu_i.shape, np.sqrt(0.1))  # Standard deviation is the sqrt of variance
        covariance_matrix = torch.full((self.action_dim, self.action_dim), 0.1)
        dist = dists.MultivariateNormal(mu_i, covariance_matrix)
        sample = dist.rsample()
        a = torch.tanh(sample) * self.action_bound
        #print(a.shape)
        #mu_i = mu[lamda_i * self.action_dim : (lamda_i + 1) * self.action_dim]
        return a

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

def evaluate_policy2(env, agent, device, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(s[np.newaxis, :]).to(device)
                a, _ = agent.actor(s).cpu().numpy()[0]
                s_next, r, dw, tr, info = env.step(a)
                done = (dw or tr)

                total_scores += r
                s = s_next
    return int(total_scores/turns)

def evaluate_policy(env, agent, device, turns = 3):
    total_scores = 0
    total_entropy = 0
    a_array = []
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(s[np.newaxis, :]).to(device)
                a, entropy = agent.actor(s)
                a_reshaped = a.reshape(-1).detach().cpu().numpy()
                a_array.append(a_reshaped)
                #print(a_reshaped.dtype)
                #print(entropy)
                s_next, r, dw, tr, info = env.step(a_reshaped)
                done = (dw or tr)

                total_scores += r
                total_entropy += entropy
                s = s_next
    if total_scores < 10:
        print(a_array)
    return int(total_scores/turns), int(total_entropy/turns)

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
#reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r

'''
def update_lamda(X, mu, k):
    n, d = X.shape
    mus = mu.reshape(k, -1)
    
    # Pre-allocate an array for PDF values (n samples by k distributions)
    pdf_values = np.zeros((n, k))
    
    # Compute PDF values in a vectorized form
    for i, mu in enumerate(mus):
        # Assuming an identity covariance matrix for each distribution
        rv = multivariate_normal(mean=mu, cov=np.eye(d)*0.1)
        pdf_values[:, i] = rv.pdf(X) * (1e10)
    
    print(pdf_values)
    # Apply softmax to PDF values to get probabilities
    # Softmax applied across each row (axis=1)
    #probabilities = np.exp(pdf_values - np.max(pdf_values, axis=1, keepdims=True))
    #probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    probabilities = pdf_values.mean(axis=0)
    probabilities /= np.sum(probabilities, axis=0, keepdims=True)
    #mean_probabilities = probabilities.mean(axis=0)
    return probabilities
'''