import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_positional_encoding(max_len))

    def _build_positional_encoding(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
        query = self.fc_query(x) 
        key = self.fc_key(x)
        value = self.fc_value(x)

        # Split heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose dimensions to apply self-attention to each head
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute self-attention scores and apply softmax
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        scores = F.softmax(scores, dim=-1)

        # Compute weighted sum of values using attention scores
        weighted_values = torch.matmul(scores, value)

        # Merge heads
        weighted_values = weighted_values.transpose(1, 2)
        weighted_values = weighted_values.reshape(batch_size, seq_len, self.hidden_dim)

        # Apply output linear layer
        out = self.fc_out(weighted_values)
        return out ### we could visualize attention!!!

class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.fc_query = nn.Linear(input_dim, hidden_dim)
        self.fc_key = nn.Linear(input_dim, hidden_dim)
        self.fc_value = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, kv):
        batch_size = q.size(0)
        q_seq_len = q.size(1)
        kv_seq_len = kv.size(1)

        # Project inputs to query, key, and value vectors
        query = self.fc_query(q)
        key = self.fc_key(kv)
        value = self.fc_value(kv)

        # Split heads
        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        # Transpose dimensions to apply cross-attention to each head
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute cross-attention scores and apply softmax
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        scores = F.softmax(scores, dim=-1)

        # Compute weighted sum of values using attention scores
        weighted_values = torch.matmul(scores, value)

        # Merge heads
        weighted_values = weighted_values.transpose(1, 2)
        weighted_values = weighted_values.reshape(batch_size, q_seq_len, self.hidden_dim)

        # Apply output linear layer
        out = self.fc_out(weighted_values)

        return out ### we could visualize attention!!!


class Attention(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim,  nhead=2):
        super(Attention, self).__init__()
        
        #self.pos_encoding_obs = PositionalEncoding(obs_dim)
        #self.pos_encoding_actions = PositionalEncoding(action_dim)
        
        self.attn_obs = SelfAttention(obs_dim, hidden_dim, nhead)
        #self.norm_obs = nn.LayerNorm(hidden_dim)
        #self.attn_action = SelfAttention(action_dim, hidden_dim, nhead)
        #self.norm_actions = nn.LayerNorm(hidden_dim)
        
        #self.cross_atten = CrossAttention(hidden_dim, hidden_dim, nhead)
        
    def forward(self, obs, actions):
        
        obs = self.pos_encoding_obs(obs)
        attn_obs = self.attn_obs(obs)
        #obs = self.norm_obs(obs + attn_obs)
        
        #actions = self.pos_encoding_actions(actions)
        #attn_actions = self.attn_action(actions)
        #actions = self.norm_actions(actions + attn_actions)
        
       # print('obs shape ',attn_obs.shape)
       # print('actions shape ',attn_actions.shape)
        
        #cross_atten = self.cross_atten(attn_obs, attn_actions)
        return attn_obs #cross_atten

    
class CriticNetwork(nn.Module):
    def __init__(self, input_enc, n_actions, obs_time_window, hidden):
        super(CriticNetwork, self).__init__()
        self.input_enc = input_enc
        self.hidden_dim = hidden
        self.obs_time_window = obs_time_window
        self.n_actions = n_actions

        self.attention_layer = Attention(self.input_enc, self.n_actions, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim * self.obs_time_window + self.n_actions, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q = nn.Linear(self.hidden_dim, 1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, states, actions, action):
        # Reshape input to seq_len timesteps with input_dim size
        states = states.view(-1, self.obs_time_window, self.input_enc)
        actions = actions.view(-1, self.obs_time_window - 1, self.n_actions) ## -1 as actions are shorter by one

        atten_output = self.attention_layer(states, actions)
        atten_output = atten_output.view(-1, self.hidden_dim * self.obs_time_window)
        out = F.relu(self.fc1(torch.cat([atten_output, action], dim=1)))
        out = F.relu(self.fc2(out))
        q = self.q(out)

        return q

class ActorNetwork(nn.Module):
    def __init__(self, input_enc, n_actions, obs_time_window, env, hidden):
        super(ActorNetwork, self).__init__()
        self.input_enc = input_enc
        self.hidden_dim = hidden
        self.obs_time_window = obs_time_window
        self.n_actions = n_actions
        self.max_action = env.action_space.high[:3]
        self.reparam_noise = 1e-6
        
        self.transformer_layer = Attention(self.input_enc, self.n_actions, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim * self.obs_time_window, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.n_actions)
        self.sigma = nn.Linear(self.hidden_dim, self.n_actions)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, states, actions):
                
        # Reshape input to seq_len timesteps with input_dim size
        states = states.view(-1, self.obs_time_window, self.input_enc)
        actions = actions.view(-1, self.obs_time_window - 1, self.n_actions) ## -1 as actions are shorter by one
        #print(actions.shape)
        atten_output = self.transformer_layer(states, actions)
        atten_output = atten_output.view(-1, self.hidden_dim * self.obs_time_window)
        #print('after', atten_output)
        prob = F.relu(self.fc1(atten_output))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, states, actions, reparameterize=True):
        
        mu, sigma = self.forward(states, actions)
        
        # Add extra noise to mu if reparameterize
        if reparameterize:
            noise = torch.randn_like(mu) * 0.2
            mu = mu + noise
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
