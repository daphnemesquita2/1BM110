# PART 2: Training the Agent
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"  

import pickle
import numpy as np
import matplotlib.pyplot as plt
from knapsack_env import BoundedKnapsackEnv        
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

SEEDS = [9451, 1697, 3229]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_part2")
os.makedirs(FIGURES_DIR, exist_ok=True)
np.save(os.path.join(SCRIPT_DIR, "part2_seeds.npy"), SEEDS)

TRAIN_TIMESTEPS = 50_000   
TUNE_TIMESTEPS  = 10_000   
DQN_BASE        = {"buffer_size": 50_000}  

np.random.seed(SEEDS[0])  
env_inspect     = BoundedKnapsackEnv(n_items=200, max_weight=200)  
state_space, _  = env_inspect.reset()                              
action_space_size = env_inspect.action_space.n                    

print(f"  State space shape : {state_space.shape}")
print(f"  Action space size : {action_space_size}")
print(f"  (3 rows: weights, values, limits  |  201 cols: 200 items + 1 knapsack info)")

# Callback: 
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True

# Helpers
def train(algo_class, algo_kwargs, seed, total_timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed)                                           
    env   = BoundedKnapsackEnv(n_items=200, max_weight=200)       
    model = algo_class("MlpPolicy", env, seed=seed, verbose=0,     
                       **algo_kwargs)
    cb = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    return cb, model

def smooth(rewards, window=20):
    if len(rewards) < window:
        return np.array(rewards)
    return np.convolve(rewards, np.ones(window) / window, mode="valid")

def aggregate(reward_lists, window=20):
    smoothed = [smooth(r, window) for r in reward_lists]
    min_len  = min(len(s) for s in smoothed)
    arr      = np.array([s[:min_len] for s in smoothed])
    return arr.mean(axis=0), arr.std(axis=0)

dqn_rewards, ppo_rewards = [], []
dqn_models,  ppo_models  = [], []

print(f"  Total runs: {len(SEEDS) * 2}  ({len(SEEDS)} seeds x 2 algorithms)")

for i, seed in enumerate(SEEDS, 1):
    print(f"\n  [{i}/{len(SEEDS)}] Seed={seed}")

    print(f"    DQN training ...", end=" ", flush=True)
    cb, m = train(DQN, DQN_BASE, seed)
    dqn_rewards.append(cb.episode_rewards)
    dqn_models.append(m)
    print(f"done  ({len(cb.episode_rewards)} episodes)")

    print(f"    PPO training ...", end=" ", flush=True)
    cb, m = train(PPO, {}, seed)
    ppo_rewards.append(cb.episode_rewards)
    ppo_models.append(m)
    print(f"done  ({len(cb.episode_rewards)} episodes)")

dqn_mean, dqn_std = aggregate(dqn_rewards)
ppo_mean, ppo_std = aggregate(ppo_rewards)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, mean, std, label, color in zip(
    axes,
    [dqn_mean, ppo_mean],
    [dqn_std,  ppo_std],
    ["DQN", "PPO"],
    ["steelblue", "darkorange"],
):
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color, label="mean")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=color, label="±1 std")
    ax.set_title(f"{label} – Training Reward (3 seeds)")
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Reward (×0.01 scaled)")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "part2_training_curves.png"), dpi=150)
plt.show()


print("Evaluating trained agents")

dqn_eval_rewards, ppo_eval_rewards = [], []
for seed, dqn_m, ppo_m in zip(SEEDS, dqn_models, ppo_models):
    np.random.seed(seed)                                          
    eval_env = BoundedKnapsackEnv(n_items=200, max_weight=200)
    m, s = evaluate_policy(dqn_m, eval_env, n_eval_episodes=10)
    dqn_eval_rewards.append(m)
    m, s = evaluate_policy(ppo_m, eval_env, n_eval_episodes=10)
    ppo_eval_rewards.append(m)

print(f"\nDQN eval — mean: {np.mean(dqn_eval_rewards):.4f}  std: {np.std(dqn_eval_rewards):.4f}  best: {np.max(dqn_eval_rewards):.4f}")
print(f"PPO eval — mean: {np.mean(ppo_eval_rewards):.4f}  std: {np.std(ppo_eval_rewards):.4f}  best: {np.max(ppo_eval_rewards):.4f}")
np.save(os.path.join(SCRIPT_DIR, "part2_ppo_eval.npy"), ppo_eval_rewards) 

def tune(algo_class, base_kwargs, param_name, values, seeds=SEEDS):
    algo_name  = algo_class.__name__
    cache_file = f"cache_tune_{algo_name}_{param_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  Testing {len(values)} values x {len(seeds)} seeds = {len(values)*len(seeds)} runs  ({TUNE_TIMESTEPS:,} steps each)")
    results = {}
    for i, val in enumerate(values, 1):
        kwargs       = {**base_kwargs, param_name: val}
        reward_lists = []
        for seed in seeds:
            print(f"    [{i}/{len(values)}] {param_name}={val}  seed={seed} ...", end=" ", flush=True)
            cb, _ = train(algo_class, kwargs, seed, total_timesteps=TUNE_TIMESTEPS)
            reward_lists.append(cb.episode_rewards)
        results[val] = aggregate(reward_lists)

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    return results

def plot_tune(results, param_name, algo_name, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, (mean, std) in results.items():
        x = np.arange(len(mean))
        ax.plot(x, mean, label=f"{param_name}={val}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(f"{algo_name} – Tuning {param_name}")
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Reward (×0.01 scaled)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.show()

print("Hyperparameter Tuning")
lr_vals = [1e-5, 1e-4, 3e-4, 1e-3, 5e-3]
plot_tune(tune(DQN, DQN_BASE, "learning_rate", lr_vals), "learning_rate", "DQN", "tune_dqn_lr.png")
plot_tune(tune(PPO, {}, "learning_rate", lr_vals), "learning_rate", "PPO", "tune_ppo_lr.png")

bs_vals = [32, 64, 128, 256, 512]
plot_tune(tune(DQN, DQN_BASE, "batch_size", bs_vals), "batch_size", "DQN", "tune_dqn_bs.png")
plot_tune(tune(PPO, {}, "batch_size", bs_vals), "batch_size", "PPO", "tune_ppo_bs.png")

plot_tune(tune(DQN, DQN_BASE, "exploration_fraction", [0.05, 0.1, 0.2, 0.4, 0.6]),
          "exploration_fraction", "DQN", "tune_dqn_ef.png")

plot_tune(tune(PPO, {}, "n_steps", [64, 128, 256, 512, 1024]),
          "n_steps", "PPO", "tune_ppo_nsteps.png")

