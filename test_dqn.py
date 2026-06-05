import argparse
import torch
from drqn_env import DRQNEnv
from utils import set_seed
from train_mlp import DDQNAgent as DDQNAgent_MLP
from train_drqn import DRQNAgent as DDQNAgent_DRQN

def parse_args():
    parser = argparse.ArgumentParser(description="Test a DDQN agent.")
    parser.add_argument("--model", type=str, required=True, choices=["MLP", "GRU", "LSTM"], help="DDQN architecture.")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for loading the trained model.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    agent = DDQNAgent_MLP(model_type=args.model, num_episodes=None) if args.model == "MLP" else DDQNAgent_DRQN(model_type=args.model, num_episodes=None)

    # Load saved model
    model_path = f"runs/{args.model}/{args.run_id}/model.pt"
    agent.online_net.load_state_dict(torch.load(model_path))
    agent.online_net.eval()

    N_ACTIONS = 8
    env = DRQNEnv(n_actions=N_ACTIONS, render_mode="gui")
    for episode in range(100):
        env.reset()
        done = False
        cumulative_reward = 0.0
        while not done:
            state = env.high_level_state()
            action = agent.select_action(state, training=False)  # Use the trained policy for action selection
            state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            cumulative_reward += reward
        print(f"Episode={episode}, reward={cumulative_reward}")
