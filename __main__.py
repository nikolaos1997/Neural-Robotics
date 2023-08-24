import numpy as np
import pandas as pd
from tqdm import tqdm
from SAC.Agent import Agent
from Robot.Environment import Env
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show   
from matplotlib import rcParams
from IPython import display
import time
import copy
import torch
import cv2
import os
from Utils import Utils


class Agent_Training(Utils):
    def __init__(self, env, subpolicy, seed = 1):
        self.env = env
        self.seed = seed
        self.static = False
        self.complete = False
        self.subpolicy = subpolicy
        self.n_epochs = 10
        self.n_eval_episodes = 5
        self.n_test_episodes = 15
        self.episode_len = 100
        self.exploration_steps = 3000
        self.seq_len = 5
        self.n_actions = self.env.action_space.shape[0]
        self.input_dims = self.env.observation_space.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent = Agent(self.input_dims, self.n_actions, self.seq_len, self.env, self.seed) 
        
        if self.subpolicy > 1: self.trained_actors = self.get_trained_actors(target_subpolicy = self.subpolicy)
     

    def train(self):
        """
        - Train the agent, save metrics for visualization, and save model
        """
        all_mean_rewards, all_success = [], []
        self.initial_exploration() 
        for epoch in range(self.n_epochs):
            seq_observation, seq_action = self.initial_window_all()  
            seq_observation_ = copy.deepcopy(seq_observation)
            for t in range(self.episode_len):
                action = self.sample_actions(seq_observation, seq_action, self.agent.actor,  reparameterize=True)
                observation_, reward, done, info = self.env.step(action, self.subpolicy, self.static) 
                seq_observation_, seq_action = self.update_sequences(seq_observation_, seq_action, observation_, action)
                # ----------- Store Transitions --------------
                self.agent.remember(seq_observation[-1], seq_action[-1], reward, seq_observation_[-1], done) #######  CHANGE HERE FOR BUFFER!!!
                actor_loss = self.agent.learn()
                seq_observation = seq_observation_
                if done: break

            mean_rewards, success = self.validate_train()
            self.agent.save_models(self.subpolicy)
            all_mean_rewards.append(mean_rewards)
            all_success.append(success)
            print(f'Epoch: {epoch}, Rewards: {mean_rewards}, Actor Loss: {actor_loss}')

            plt.plot(all_mean_rewards)
            plt.savefig(f'Figures/Figures_sub_{self.subpolicy}.png', dpi=300, bbox_inches='tight')
            
        # Save data as text file
        all_data = np.array([all_mean_rewards, all_success]).astype(float)
        all_data = all_data.T
        with open(f'Data/Data_sub_{self.subpolicy}.txt', 'w') as file:
            file.write('Rewards \tSuccess\n')  # Write column headers
            np.savetxt(file, all_data, delimiter='\t', fmt='%f')
                       
                
    def validate_train(self):
        """
        - Validate the agent during training in every epoch
        """
        total_rewards = 0
        success = 0
        actor = self.agent.actor.eval()
        for _ in range(self.n_eval_episodes):
            episode_reward = 0
            seq_observation, seq_action = self.initial_window_all()
            seq_observation_ = copy.deepcopy(seq_observation)
            for t in range(self.episode_len):
                action = self.sample_actions(seq_observation, seq_action, actor, reparameterize=False)
                observation_, reward, done, info = self.env.step(action, self.subpolicy, self.static)
                if info: 
                    print('--------------- Succesful ---------------')
                    success +=1
                episode_reward += reward
                seq_observation_, seq_action = self.update_sequences(seq_observation_, seq_action, observation_, action)
                seq_observation = seq_observation_
                if done: break
            total_rewards += episode_reward
        return total_rewards / self.n_eval_episodes, (success / self.n_eval_episodes)*100

if __name__ == '__main__':
    
    env = Env(seed = 1)
    m = Agent_Training(env, subpolicy = 2)
    m.train()