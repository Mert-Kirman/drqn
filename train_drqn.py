import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from drqn_env import DRQNEnv
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PROPER DRQN agent.")
    parser.add_argument("--model", type=str, required=True, choices=["GRU", "LSTM"], help="RNN architecture.")
    parser.add_argument("--num_episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class EpisodicReplayBuffer:
    def __init__(self, capacity):
        # Capacity is now measured in total EPISODES, not individual steps
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, episode_transitions):
        # episode_transitions is a list of tuples: (state, action, reward, next_state, terminated)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode_transitions
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)

class StatefulQNetwork(nn.Module):
    def __init__(self, model_type, state_dim=6, n_actions=8, hidden_dim=128):
        super(StatefulQNetwork, self).__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        self.norm = nn.LayerNorm(state_dim)
        
        if model_type == "GRU":
            self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(state_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, state_dim)
        x = self.norm(x)
        out, new_hidden = self.rnn(x, hidden) # out shape: (batch_size, seq_len, hidden_dim)
        # We output the Q-values for all sequence steps for BPTT
        q_values = self.fc(out) # shape: (batch_size, seq_len, n_actions)
        return q_values, new_hidden
    
    def init_hidden(self, batch_size=1, device="cpu"):
        # Helper function to generate clean hidden states
        if self.model_type == "GRU":
            return torch.zeros(1, batch_size, self.hidden_dim, device=device) # shape: (num_layers * num_directions, batch_size, hidden_size)
        else: # LSTM requires both hidden and cell states
            return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                    torch.zeros(1, batch_size, self.hidden_dim, device=device))

class DRQNAgent:
    def __init__(self, model_type, num_episodes, state_dim=6, n_actions=8):
        self.n_actions = n_actions
        self.model_type = model_type
        self.device = torch.device("cpu")
        
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 20000 
        self.tau = 0.005
        self.batch_size = 32  # Batch size is smaller because we are sampling 32 WHOLE episodes!
        self.learning_rate = 0.0001
        self.update_freq = 4  
        
        # Buffer tracks 2000 episodes (~100,000 steps). Warm up is 100 episodes.
        self.learning_starts_episodes = 100 
        self.buffer_capacity = 2000
        self.memory = EpisodicReplayBuffer(self.buffer_capacity)
        self.steps_done = 0
        
        self.online_net = StatefulQNetwork(model_type, state_dim, n_actions).to(self.device)
        self.target_net = StatefulQNetwork(model_type, state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none') # We manage reduction manually using the mask
        
    def select_action(self, state, hidden, training=True):
        if training:
            if len(self.memory) >= self.learning_starts_episodes:
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
            else:
                epsilon = 1.0
            
            if random.random() < epsilon:
                return random.randint(0, self.n_actions - 1), hidden
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, state_dim)
            q_values, new_hidden = self.online_net(state_tensor, hidden)
            action = q_values.argmax().item()
            return action, new_hidden

    def optimize_model(self):
        if len(self.memory) < self.learning_starts_episodes:
            return
        
        episodes = self.memory.sample(self.batch_size)
        
        # Determine the longest episode in the batch for padding
        max_len = max(len(ep) for ep in episodes)
        
        # Initialize padded tensors
        states = torch.zeros(self.batch_size, max_len, 6).to(self.device)
        next_states = torch.zeros(self.batch_size, max_len, 6).to(self.device)
        actions = torch.zeros(self.batch_size, max_len, 1, dtype=torch.long).to(self.device)
        rewards = torch.zeros(self.batch_size, max_len, 1).to(self.device)
        terminated = torch.zeros(self.batch_size, max_len, 1).to(self.device)
        mask = torch.zeros(self.batch_size, max_len, 1).to(self.device)
        
        # Populate the tensors
        for b, episode in enumerate(episodes):
            length = len(episode)
            s, a, r, ns, d = zip(*episode)
            states[b, :length] = torch.FloatTensor(np.array(s))
            next_states[b, :length] = torch.FloatTensor(np.array(ns))
            actions[b, :length] = torch.LongTensor(a).unsqueeze(1)
            rewards[b, :length] = torch.FloatTensor(r).unsqueeze(1)
            terminated[b, :length] = torch.FloatTensor(d).unsqueeze(1)
            mask[b, :length] = 1.0  # Boolean mask to ignore the zero-padded steps
            
        # Get Q-values for the entire sequence at once
        hidden = self.online_net.init_hidden(self.batch_size, self.device)
        all_q_values, _ = self.online_net(states, hidden)
        q_values = all_q_values.gather(2, actions)
        
        # Calculate Target Q-values
        with torch.no_grad():
            target_hidden = self.target_net.init_hidden(self.batch_size, self.device)
            # Find best actions using online net
            all_next_q_online, _ = self.online_net(next_states, hidden)
            best_next_actions = all_next_q_online.argmax(dim=2, keepdim=True)
            
            # Evaluate with target net
            all_next_q_target, _ = self.target_net(next_states, target_hidden)
            next_q_values = all_next_q_target.gather(2, best_next_actions)
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terminated))

        # Calculate masked loss
        loss = self.criterion(q_values, target_q_values)
        masked_loss = (loss * mask).sum() / mask.sum() # Average only over valid steps

        self.optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.soft_update()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        online_net_state_dict = self.online_net.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{args.model}", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.pt")

    env = DRQNEnv(n_actions=8, render_mode="blind")
    agent = DRQNAgent(model_type=args.model, num_episodes=args.num_episodes, state_dim=6, n_actions=8)
    
    episode_rewards, episode_rps, episode_final_dists = [], [], []
    best_avg_dist = float('inf') 
    
    sys.stdout = Logger(os.path.join(run_dir, 'train_log.txt'))
    
    global_step = 0
    
    for episode in tqdm(range(args.num_episodes), desc=f"Training DRQN with {args.model}"):
        env.reset()
        current_state = env.high_level_state()
        hidden_state = agent.online_net.init_hidden(batch_size=1, device=agent.device)
        
        episode_transitions = []
        cumulative_reward = 0.0
        done = False
        
        while not done:
            # 1-frame input, but we pass and receive the hidden state
            action, next_hidden = agent.select_action(current_state, hidden_state)
            _, _, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            # The environment already contains the updated reward shaping
            reward = env.reward() 
            next_state = env.high_level_state()
            
            episode_transitions.append((current_state, action, reward, next_state, terminated))
            
            current_state = next_state
            hidden_state = next_hidden
            cumulative_reward += reward
            global_step += 1
            
            if global_step % agent.update_freq == 0:
                agent.optimize_model()
                
        # Push the entire episode into the buffer at once
        agent.memory.push(episode_transitions)
        
        episode_rewards.append(cumulative_reward)
        episode_rps.append(cumulative_reward / max(len(episode_transitions), 1))
        
        final_state = env.high_level_state()
        final_dist = np.linalg.norm(final_state[2:4] - final_state[4:6])
        episode_final_dists.append(final_dist)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rps = np.mean(episode_rps[-100:])
            avg_dist = np.mean(episode_final_dists[-100:])
            tqdm.write(f"Episode {episode+1} | Avg Reward: {avg_reward:.2f} | Avg RPS: {avg_rps:.2f} | Avg Final Distance: {avg_dist:.4f}")

            # Guard: Only allow saving if training has actually begun
            if len(agent.memory) >= agent.learning_starts_episodes:
                if avg_dist < best_avg_dist:
                    best_avg_dist = avg_dist
                    torch.save(agent.online_net.state_dict(), model_path)
                    tqdm.write(f"*** New Best {args.model} Model Saved (Avg Final Distance: {best_avg_dist:.4f}) ***")
    
    # Save training metrics as numpy arrays
    rewards_path = os.path.join(run_dir, "rewards.npy")
    rps_path = os.path.join(run_dir, "rps.npy")
    dists_path = os.path.join(run_dir, "final_dists.npy")
    np.save(rewards_path, np.array(episode_rewards))
    np.save(rps_path, np.array(episode_rps))
    np.save(dists_path, np.array(episode_final_dists))
    print(f"Training metrics saved")

    # Save hyperparameters and results
    hyperparams_path = os.path.join(run_dir, "config.txt")
    final_avg_reward = np.mean(episode_rewards[-100:])
    final_avg_rps = np.mean(episode_rps[-100:])
    final_avg_dist = np.mean(episode_final_dists[-100:])
    with open(hyperparams_path, 'w') as f:
        f.write(f"Training Run ({args.model}): {timestamp}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  model_type: {args.model}\n")
        f.write(f"  gamma: {agent.gamma}\n")
        f.write(f"  eps_start: {agent.eps_start}\n")
        f.write(f"  eps_end: {agent.eps_end}\n")
        f.write(f"  eps_decay: {agent.eps_decay}\n")
        f.write(f"  tau: {agent.tau}\n")
        f.write(f"  batch_size: {agent.batch_size}\n")
        f.write(f"  learning_rate: {agent.learning_rate} (Adam, Constant)\n")
        f.write(f"  update_freq: {agent.update_freq}\n")
        f.write(f"  buffer_capacity (Episodes): {agent.buffer_capacity}\n")
        f.write(f"  num_episodes: {args.num_episodes}\n")
        f.write(f"\nResults:\n")
        f.write(f"  final_avg_reward (last 100): {final_avg_reward:.2f}\n")
        f.write(f"  final_avg_rps (last 100): {final_avg_rps:.2f}\n")
        f.write(f"  max_reward: {max(episode_rewards):.2f}\n")
        f.write(f"  min_reward: {min(episode_rewards):.2f}\n")
        f.write(f"  final_avg_dist (last 100): {final_avg_dist:.4f}\n")
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Plot Results
    figure_path = os.path.join(run_dir, "training_plot.png")
    plt.figure(figsize=(8, 12))
    
    # Plot 1: Cumulative Reward
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, alpha=0.6)
    window = min(100, len(episode_rewards))
    moving_avg_reward = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), moving_avg_reward, label=f'{window}-Episode Moving Avg')
    plt.title(f'{args.model} Cumulative Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot 2: Reward Per Step (RPS)
    plt.subplot(3, 1, 2)
    plt.plot(episode_rps, alpha=0.6)
    moving_avg_rps = np.convolve(episode_rps, np.ones(100)/100, mode='valid')
    plt.plot(moving_avg_rps, label='100-Episode Moving Avg')
    plt.title(f'{args.model} Reward Per Step (RPS) over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('RPS')
    plt.legend()
    
    # Plot 3: Final Object-to-Goal Distance (Lower is Better)
    plt.subplot(3, 1, 3)
    plt.plot(episode_final_dists, alpha=0.3, color='red')
    moving_avg_dist = np.convolve(episode_final_dists, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_final_dists)), moving_avg_dist, color='darkred', label=f'{window}-Episode Moving Avg')
    plt.title(f'{args.model} Final Object-to-Goal Distance (Lower is Better)')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(figure_path)
    print(f"Training complete! Plot saved to {figure_path}")
