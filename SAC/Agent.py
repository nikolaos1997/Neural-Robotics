import os
import torch as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy
from SAC.Buffer import ReplayBuffer
from SAC.Networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, input_dims, n_actions, seq_len, env, seed, attention_heads = 4, lr = 0.0002, #0.0004,  
            gamma=0.99, max_size=100000, tau=0.005, batch_size=460, reward_scale=1):
        self.gamma = gamma
        self.seed = seed
        self.tau = tau
        self.reward_scale = reward_scale
        self.memory = ReplayBuffer(max_size, input_dims, n_actions, seq_len)
        self.batch_size = batch_size
        self.env = env
        self.lr = lr
        self.attention_heads = attention_heads
        self.alpha_autotune = False
        
        # Set random seeds
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        
        # Networks
        self.actor = ActorNetwork(self.lr, input_dims, n_actions, env, seq_len, self.attention_heads)
        self.critic_1 = CriticNetwork(self.lr, input_dims, seq_len, n_actions, self.attention_heads)
        self.critic_2 = CriticNetwork(self.lr, input_dims, seq_len, n_actions, self.attention_heads)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        #Optimizers
        self.actor_optim = T.optim.Adam(self.actor.parameters(), lr = self.lr)
        self.critic_1_optim = T.optim.Adam(self.critic_1.parameters(), lr = self.lr)
        self.critic_2_optim = T.optim.Adam(self.critic_2.parameters(), lr = self.lr)
        
        # Entropy 
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = T.nn.Parameter(T.zeros(1, requires_grad=True).to(self.actor.device))
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr = self.lr)
        
    def remember(self, state, seq_action, reward, new_state, done):
        self.memory.store_transition(state, seq_action, reward, new_state, done)
        
        # update the target network
    def update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def save_models(self, subpolicy):
        print('.... saving models ....')
        T.save(self.actor.state_dict(), f'Models/models_sub_{subpolicy}/Actor')
            
    
    def learn(self):
        
        state, action_seq, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        inverse_done = T.tensor(1 - done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action_seq = T.tensor(action_seq, dtype=T.float).to(self.actor.device)

        last_action = action_seq[:,-1] # for the value net training
        seq_actions_ = action_seq[:,1:] # to train along the next observations
        seq_actions = action_seq[:,:-1] # to train along the observations
       
      # ACTOR AND ALPHA UPDATE ----------------------------
        action, log_probs = self.actor.sample_normal(state, seq_actions, reparameterize=True)
        log_probs = log_probs.view(-1)
        if self.alpha_autotune: 
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = self.log_alpha.exp()
            
        else: alpha = 0.001
        
        q1_new_policy = self.critic_1.forward(state, seq_actions, action)
        q2_new_policy = self.critic_2.forward(state, seq_actions, action)
        actor_loss = T.mean(log_probs * alpha - T.min(q1_new_policy, q2_new_policy).view(-1))
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        ## Q VALUE UPDATE --------------------------------
        
        q1_value = self.critic_1.forward(state, seq_actions, last_action).view(-1)
        q2_value = self.critic_2.forward(state, seq_actions, last_action).view(-1)
        with T.no_grad():
            action_, log_probs_ = self.actor.sample_normal(state_, seq_actions_, reparameterize=True)
            log_probs_ = log_probs_.view(-1)
            target_q1_value = self.target_critic_1.forward(state_, seq_actions_, action_)
            target_q2_value = self.target_critic_2.forward(state_, seq_actions_, action_)
            target_q_value = T.min(target_q1_value, target_q2_value).view(-1) - alpha * log_probs_
            final_target = self.reward_scale * reward + inverse_done * self.gamma * target_q_value
        
        # CRITIC LOSS 1
        q1_loss = 0.5 * F.mse_loss(q1_value, final_target)
        self.critic_1_optim.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.critic_1_optim.step()
                
        # CRITIC LOSS 2
        q2_loss = 0.5 * F.mse_loss(q2_value, final_target)
        self.critic_2_optim.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.critic_2_optim.step()
        
        self.update_target_network(self.target_critic_1, self.critic_1)
        self.update_target_network(self.target_critic_2, self.critic_2)
        
        return actor_loss.item()#, q1_loss.item(), q2_loss.item()#, alpha.item(), alpha_loss.item()
