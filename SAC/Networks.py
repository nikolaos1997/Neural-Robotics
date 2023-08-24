import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_positional_encoding(seq_len))

    def _build_positional_encoding(self, seq_len):
        pe = T.zeros(seq_len, self.d_model)
        position = T.arange(0, seq_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, self.d_model).float() * (-math.log(10000.0) / self.d_model))
        div_term = div_term.view(1, -1).repeat(seq_len, 1)
        pe[:, 0::2] = T.sin(position * div_term[:, 0::2])
        pe[:, 1::2] = T.cos(position * div_term[:, 1::2])
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.fc_query = nn.Linear(input_dim, hidden_dim)
        self.fc_key = nn.Linear(input_dim, hidden_dim)
        self.fc_value = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Project inputs to query, key, and value vectors
        query = self.fc_query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.fc_key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.fc_value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute self-attention scores and apply softmax
        scores = T.matmul(query, key.transpose(-2, -1)) / T.sqrt(T.tensor(self.head_dim).float())
        scores = F.softmax(scores, dim=-1)

        # Compute weighted sum of values using attention scores
        weighted_values = T.matmul(scores, value).transpose(1, 2)
        weighted_values = weighted_values.reshape(batch_size, seq_len, self.hidden_dim)
        out = self.fc_out(weighted_values)

        return out
    
class Attention(nn.Module):
    def __init__(self, obs_dims, action_dims, seq_len, nhead, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        #self.position_enc = PositionalEncoding(obs_dims + action_dims, seq_len)
        self.self_attn_obs = SelfAttention(obs_dims + action_dims, hidden_dim, nhead)
        self.fc = nn.Linear(hidden_dim * seq_len, hidden_dim * 2)
                
    def forward(self, obs, actions):
        actions = F.pad(actions, (0, 0, 0, 1), "constant", 0)
        comb = T.cat([obs, actions], dim=-1)
        
        #comb = self.position_enc(comb)
        attn = self.self_attn_obs(comb)
        attn = attn.view(-1, self.hidden_dim * self.seq_len)
        out = F.relu(self.fc(attn))
        return out

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, seq_len, n_actions, num_heads):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = 256
        self.seq_len = seq_len
        self.n_actions = n_actions
        self.num_heads = num_heads
        
        self.attention_layer = Attention(self.input_dims, self.n_actions, seq_len, self.num_heads,  self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim * 2 + self.n_actions, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q = nn.Linear(self.hidden_dim, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, seq_actions, action):

        # Apply self-attention to each timestep
        attention = self.attention_layer(state, seq_actions)
        
        # Apply fully connected layer and activation function
        out = F.relu(self.fc1(T.cat([attention, action], dim=1)))
        out = F.relu(self.fc2(out))
        q = self.q(out)
        return q

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims,n_actions, env, seq_len, num_heads):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = 256
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.n_actions = n_actions
        self.max_action = env.action_space.high[:3]
        self.reparam_noise = 1e-6
        
        self.attention_layer = Attention(self.input_dims, self.n_actions, seq_len, self.num_heads,  self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.n_actions)
        self.sigma = nn.Linear(self.hidden_dim, self.n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, seq_actions):

        # Apply self-attention to each timestep
        attention = self.attention_layer(state, seq_actions)
        
        # Apply fully connected layer and activation function
        prob = F.relu(self.fc1(attention))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, seq_actions, reparameterize=True):
        
        mu, sigma = self.forward(state, seq_actions)
        
        # Add extra noise to mu if reparameterize
        if reparameterize:
            noise = T.randn_like(mu) * 0.2
            mu = mu + noise
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

