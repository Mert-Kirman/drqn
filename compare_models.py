import os
import numpy as np
import matplotlib.pyplot as plt

runs_to_compare = {
    "old_mlp": "runs/old_mlp/run_20260608_144047",
    "MLP": "runs/MLP/run_20260607_185121",
    "GRU": "runs/GRU/run_20260607_184743",
    "LSTM": "runs/LSTM/run_20260607_185120"
}

run_results = {}

for label, run_dir in runs_to_compare.items():
    rewards_path = os.path.join(run_dir, "rewards.npy")
    rps_path = os.path.join(run_dir, "rps.npy")
    dists_path = os.path.join(run_dir, "final_dists.npy")
    
    # Check if all three metric files exist for the given run
    if os.path.exists(rewards_path) and os.path.exists(rps_path) and os.path.exists(dists_path):
        rewards = np.load(rewards_path)
        rps = np.load(rps_path)
        dists = np.load(dists_path)
        
        run_results[label] = {
            "rewards": rewards,
            "rps": rps,
            "dists": dists
        }
    else:
        print(f"Metrics not found for {label} at directory: {run_dir}")

# Create a composite figure with 3 subplots
plt.figure(figsize=(12, 12)) 
window_size = 100

for label, metrics in run_results.items():
    rewards = metrics["rewards"]
    rps = metrics["rps"]
    dists = metrics["dists"]
    
    # Compute moving averages to smooth the plots
    rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    rps_ma = np.convolve(rps, np.ones(window_size)/window_size, mode='valid')
    dists_ma = np.convolve(dists, np.ones(window_size)/window_size, mode='valid')
    
    # 1. Plot Reward
    plt.subplot(3, 1, 1)
    plt.plot(rewards_ma, label=label, linewidth=1.5)
    
    # 2. Plot RPS
    plt.subplot(3, 1, 2)
    plt.plot(rps_ma, label=label, linewidth=1.5)
    
    # 3. Plot Final Distance
    plt.subplot(3, 1, 3)
    plt.plot(dists_ma, label=label, linewidth=1.5)

# --- Format Subplot 1 (Reward) ---
plt.subplot(3, 1, 1)
plt.title(f"Cumulative Reward ({window_size}-Episode Moving Average)", fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(alpha=0.4)
plt.legend(fontsize=11)

# --- Format Subplot 2 (RPS) ---
plt.subplot(3, 1, 2)
plt.title(f"Reward Per Step ({window_size}-Episode Moving Average)", fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('RPS', fontsize=12)
plt.grid(alpha=0.4)
plt.legend(fontsize=11)

# --- Format Subplot 3 (Final Distance) ---
plt.subplot(3, 1, 3)
plt.title(f"Final Object-to-Goal Distance ({window_size}-Episode Moving Average)", fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Distance (meters)', fontsize=12)
# Draw a horizontal line at the goal threshold (0.05m) to visualize success
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.6, label='Goal Threshold (0.05m)')
plt.grid(alpha=0.4)
plt.legend(fontsize=11)

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
save_path = "assets/model_comparisons.png"
plt.savefig(save_path, dpi=300)
print(f"Comparison plot saved successfully to {save_path}")
