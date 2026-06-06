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
from collections import deque
from tqdm import tqdm

from drqn_env import DRQNEnv
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDQN agent.")
    parser.add_argument("--model", type=str, default="MLP", choices=["MLP"], help="Which model architecture to use for the Q-network.")
    parser.add_argument("--num_episodes", type=int, default=4000, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_seq, action, reward, next_state_seq, terminated):
        self.buffer.append((state_seq, action, reward, next_state_seq, terminated))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, model_type, seq_len=4, state_dim=6, n_actions=8, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.model_type = model_type
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        if model_type == "MLP":
            # Flatten the sequence for the MLP baseline (4 frames * 6 dims = 24)
            self.network = nn.Sequential(
                nn.Linear(seq_len * state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )

    def forward(self, x):
        # x shape: (batch_size, seq_len, state_dim)
        if self.model_type == "MLP":
            x = x.view(x.size(0), -1) 
            return self.network(x)

class DDQNAgent:
    def __init__(self, model_type, num_episodes, seq_len=4, state_dim=6, n_actions=8):
        self.n_actions = n_actions
        self.seq_len = seq_len
        
        self.device = torch.device("cpu")
        print(f"Training on device: {self.device}")
        
        # Hyperparameters
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 40000
        self.tau = 0.005
        self.batch_size = 256
        self.learning_rate = 0.0001
        self.update_freq = 4  
        self.learning_starts = 4000  # Warm-up period before training starts
        self.buffer_capacity = 100000
        self.memory = ReplayBuffer(self.buffer_capacity)
        
        self.steps_done = 0
        
        self.online_net = QNetwork(model_type, seq_len, state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(model_type, seq_len, state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss() 
        
    def select_action(self, state_seq, training=True):
        if training:
            # Only decay epsilon if the warm-up period is over
            if len(self.memory) >= self.learning_starts:
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
            else:
                epsilon = 1.0 # Force 100% random exploration during warm-up
            
            if random.random() < epsilon:
                return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device) # (1, seq_len, state_dim)
            return self.online_net(state_tensor).argmax().item()

    def optimize_model(self):
        # Block updates until the buffer has enough diverse data to sample a meaningful batch
        if len(self.memory) < self.learning_starts:
            return
        
        states, actions, rewards, next_states, terminated = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)  # (batch_size, seq_len, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        terminated = torch.FloatTensor(terminated).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.online_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            # Online network selects the best action for the next state
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            
            # Target network evaluates the Q-value of that specific action
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            
            # Calculate the Bellman target
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terminated))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update the target network
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
    
    seq_len = 4
    agent = DDQNAgent(model_type=args.model, num_episodes=args.num_episodes, seq_len=seq_len, state_dim=6, n_actions=8)
    
    episode_rewards = []
    episode_rps = [] # Reward Per Step
    episode_final_dists = [] # Track the final Object-to-Goal distance
    best_avg_dist = float('inf') # We want to minimize the distance
    
    sys.stdout = Logger(os.path.join(run_dir, 'train_log.txt'))
    for episode in tqdm(range(args.num_episodes), desc=f"Training DDQN with {args.model}"):
        env.reset()
        current_state = env.high_level_state()
        
        # Initialize the POMDP queue by padding the first state 4 times
        state_queue = deque([current_state] * seq_len, maxlen=seq_len)
        
        cumulative_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Convert current queue to sequence array
            current_seq_array = np.array(state_queue) # shape: (seq_len, state_dim)
            
            action = agent.select_action(current_seq_array)
            _, reward, terminated, truncated = env.step(action)
            
            # The Episode loop ends on either termination or truncation
            done = terminated or truncated
            
            next_state = env.high_level_state()
            
            # Slide the window forward
            state_queue.append(next_state)
            next_seq_array = np.array(state_queue)
            
            # Pass only "terminated" to the buffer since "truncated" is just a time limit and not a true terminal state
            agent.memory.push(current_seq_array, action, reward, next_seq_array, terminated)
            
            cumulative_reward += reward
            episode_steps += 1
            
            if episode_steps % agent.update_freq == 0:
                agent.optimize_model()
            
        episode_rewards.append(cumulative_reward)
        episode_rps.append(cumulative_reward / max(episode_steps, 1))
        
        # Calculate final object-to-goal distance from the terminal state [o_y, o_z, g_y, g_z]
        final_dist = env.raw_object_goal_distance()
        episode_final_dists.append(final_dist)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rps = np.mean(episode_rps[-100:])
            avg_dist = np.mean(episode_final_dists[-100:])
            tqdm.write(f"Episode {episode+1} | Avg Reward (last 100): {avg_reward:.2f} | Avg RPS: {avg_rps:.2f} | Avg Final Distance (last 100): {avg_dist:.4f}")

            # Save the model if it achieves a new low score in final object-to-goal distance
            if len(agent.memory) >= agent.learning_starts:
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
        f.write(f"  sequence_length: {seq_len}\n")
        f.write(f"  gamma: {agent.gamma}\n")
        f.write(f"  eps_start: {agent.eps_start}\n")
        f.write(f"  eps_end: {agent.eps_end}\n")
        f.write(f"  eps_decay: {agent.eps_decay}\n")
        f.write(f"  tau: {agent.tau}\n")
        f.write(f"  batch_size: {agent.batch_size}\n")
        f.write(f"  learning_rate: {agent.learning_rate} (Adam, Constant)\n")
        f.write(f"  update_freq: {agent.update_freq}\n")
        f.write(f"  buffer_capacity: {agent.buffer_capacity}\n")
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
