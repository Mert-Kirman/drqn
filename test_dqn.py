import argparse
import time
import torch
import numpy as np
from collections import deque

from drqn_env import DRQNEnv
from utils import set_seed
from train_old_mlp import DDQNAgent as DDQNAgent_OldMLP
from train_mlp import DDQNAgent as DDQNAgent_MLP
from train_drqn import DRQNAgent as DDQNAgent_DRQN

def parse_args():
    parser = argparse.ArgumentParser(description="Test a DDQN agent.")
    parser.add_argument("--model", type=str, required=True, choices=["old_mlp", "MLP", "GRU", "LSTM"], help="DDQN architecture.")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for loading the trained model (e.g., run_20260605_153000).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Initialize the correct agent
    if args.model == "old_mlp":
        agent = DDQNAgent_OldMLP(state_dim=6, n_actions=8)
    elif args.model == "MLP":
        # The MLP requires the seq_len parameter to build the 24-dim input
        agent = DDQNAgent_MLP(model_type=args.model, num_episodes=1, seq_len=4, state_dim=6, n_actions=8)
    else:
        agent = DDQNAgent_DRQN(model_type=args.model, num_episodes=1, state_dim=6, n_actions=8)

    model_path = f"runs/{args.model}/{args.run_id}/model.pt"
        
    print(f"Loading weights from: {model_path}")
    
    # map_location ensures it loads cleanly regardless of where it was trained
    agent.online_net.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
    agent.online_net.eval()

    # Setup the Visual Environment
    N_ACTIONS = 8
    env = DRQNEnv(n_actions=N_ACTIONS, render_mode="gui")
    
    for episode in range(10):
        env.reset()
        current_state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        
        # --- ARCHITECTURE SETUP ---
        if args.model == "old_mlp":
            pass
        elif args.model == "MLP":
            # Initialize the 4-frame queue
            state_queue = deque([current_state] * agent.seq_len, maxlen=agent.seq_len)
        else:
            # Initialize the blank memory
            hidden_state = agent.online_net.init_hidden(batch_size=1, device=agent.device)
            
        while not done:
            # --- ACTION SELECTION ---
            if args.model == "old_mlp":
                state = env.high_level_state()
                action = agent.select_action(state, training=False)
            elif args.model == "MLP":
                current_seq_array = np.array(state_queue)
                action = agent.select_action(current_seq_array, training=False)
            else:
                action, next_hidden = agent.select_action(current_state, hidden_state, training=False)
                hidden_state = next_hidden
            
            # --- PHYSICS STEP ---
            _, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            
            next_state = env.high_level_state()
            
            # --- STATE UPDATE ---
            if args.model == "MLP":
                state_queue.append(next_state)
            else:
                current_state = next_state
                
            # # Slow down the loop slightly so the MuJoCo GUI is watchable
            # time.sleep(0.03)
            
        print(f"Episode {episode+1} | Total Reward: {cumulative_reward:.2f}")
