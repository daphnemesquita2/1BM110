##########################################

#### PART 2: Training the Agent ##########

##########################################

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress TensorFlow oneDNN info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"  # suppress all TensorFlow logging

import pickle
import numpy as np
import matplotlib.pyplot as plt
from knapsack_env import BoundedKnapsackEnv          # requirement 4
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ──────────────────────────────────────────────
# Seeds (requirement 2)
# Three seeds are randomly generated and printed
# so results can be reproduced. The same seeds
# are used for all algorithms (requirement: same
# 3 seeds for both DQN and PPO).
# ──────────────────────────────────────────────

SEEDS = [int(x) for x in np.random.randint(0, 10_000, size=3)]
print(f"Using seeds: {SEEDS}  (record these to reproduce results)")

# Save seeds next to this script so Part 3 can find them regardless of working directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_part2")
os.makedirs(FIGURES_DIR, exist_ok=True)
np.save(os.path.join(SCRIPT_DIR, "part2_seeds.npy"), SEEDS)


# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────

TRAIN_TIMESTEPS = 50_000   # required by assignment
TUNE_TIMESTEPS  = 10_000   # reduced for tuning — relative comparison only
DQN_BASE        = {"buffer_size": 50_000}  # prevents ~9.66GB replay buffer


# ──────────────────────────────────────────────
# Requirement 6: Inspect environment
# ──────────────────────────────────────────────

np.random.seed(SEEDS[0])  # requirement 2: set seed before env creation
env_inspect     = BoundedKnapsackEnv(n_items=200, max_weight=200)  # requirement 5
state_space, _  = env_inspect.reset()                              # requirement 6
action_space_size = env_inspect.action_space.n                     # requirement 6

print("\n========================================")
print("Environment Inspection")
print("========================================")
print(f"  State space shape : {state_space.shape}")
print(f"  Action space size : {action_space_size}")
print(f"  (3 rows: weights, values, limits  |  201 cols: 200 items + 1 knapsack info)")


# ──────────────────────────────────────────────
# Callback: record cumulative reward per episode
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def train(algo_class, algo_kwargs, seed, total_timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed)                                            # requirement 2: set seed
    env   = BoundedKnapsackEnv(n_items=200, max_weight=200)        # requirement 5
    model = algo_class("MlpPolicy", env, seed=seed, verbose=0,     # requirement 3: agent seed
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


# ──────────────────────────────────────────────
# PART A: Train DQN & PPO — 3 seeds, 50k steps
# ──────────────────────────────────────────────

dqn_rewards, ppo_rewards = [], []
dqn_models,  ppo_models  = [], []

print("\n========================================")
print("PART A: Training DQN & PPO (3 seeds, 50k steps each)")
print(f"  Total runs: {len(SEEDS) * 2}  ({len(SEEDS)} seeds x 2 algorithms)")
print("========================================")

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

print("\nAll seeds trained.")


# ── Plot training curves (aggregated mean ± std) ──

print("\n----------------------------------------")
print("Plotting training curves (mean ± std across 3 seeds) ...")
print("----------------------------------------")

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
print("\n>>> Showing training curves — close the plot window to continue ...")
plt.show()


# ── Evaluate trained agents (all 3 seeds, aggregate) ──

print("\n----------------------------------------")
print("Evaluating trained agents (10 episodes per seed, all 3 seeds) ...")
print("----------------------------------------")

dqn_eval_rewards, ppo_eval_rewards = [], []
for seed, dqn_m, ppo_m in zip(SEEDS, dqn_models, ppo_models):
    np.random.seed(seed)                                           # requirement 2
    eval_env = BoundedKnapsackEnv(n_items=200, max_weight=200)
    m, s = evaluate_policy(dqn_m, eval_env, n_eval_episodes=10)
    dqn_eval_rewards.append(m)
    m, s = evaluate_policy(ppo_m, eval_env, n_eval_episodes=10)
    ppo_eval_rewards.append(m)

print(f"\nDQN eval — mean: {np.mean(dqn_eval_rewards):.4f}  std: {np.std(dqn_eval_rewards):.4f}  best: {np.max(dqn_eval_rewards):.4f}")
print(f"PPO eval — mean: {np.mean(ppo_eval_rewards):.4f}  std: {np.std(ppo_eval_rewards):.4f}  best: {np.max(ppo_eval_rewards):.4f}")
print(f"\n(Reminder: multiply by 100 for real reward — e.g. 0.93 → 93)")
np.save(os.path.join(SCRIPT_DIR, "part2_ppo_eval.npy"), ppo_eval_rewards)  # saved so Part 3 can compare against PPO


# ──────────────────────────────────────────────
# PART B: Hyperparameter Tuning
#
#  Shared (both DQN & PPO):
#    1. learning_rate
#    2. batch_size
#  DQN-specific:
#    3. exploration_fraction
#  PPO-specific:
#    4. n_steps
# ──────────────────────────────────────────────

def tune(algo_class, base_kwargs, param_name, values, seeds=SEEDS):
    algo_name  = algo_class.__name__
    cache_file = f"cache_tune_{algo_name}_{param_name}.pkl"

    if os.path.exists(cache_file):
        print(f"  Loaded from cache: {cache_file}  (delete file to retune)")
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
            print("done")
        results[val] = aggregate(reward_lists)

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved to cache: {cache_file}")
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
    print(f"\n>>> Showing {algo_name} – {param_name} tuning plot — close the plot window to continue ...")
    plt.show()


print("\n========================================")
print("PART B: Hyperparameter Tuning")
print("  4 hyperparameters: 2 shared + 1 DQN-specific + 1 PPO-specific")
print(f"  Using same seeds as Part A: {SEEDS}")
print("========================================")

# 1. learning_rate (shared)
lr_vals = [1e-5, 1e-4, 3e-4, 1e-3, 5e-3]
print("\n[1/4] Tuning learning_rate for DQN  (shared hyperparameter) ...")
plot_tune(tune(DQN, DQN_BASE, "learning_rate", lr_vals), "learning_rate", "DQN", "tune_dqn_lr.png")
print("\n[1/4] Tuning learning_rate for PPO  (shared hyperparameter) ...")
plot_tune(tune(PPO, {}, "learning_rate", lr_vals), "learning_rate", "PPO", "tune_ppo_lr.png")

# 2. batch_size (shared)
bs_vals = [32, 64, 128, 256, 512]
print("\n[2/4] Tuning batch_size for DQN  (shared hyperparameter) ...")
plot_tune(tune(DQN, DQN_BASE, "batch_size", bs_vals), "batch_size", "DQN", "tune_dqn_bs.png")
print("\n[2/4] Tuning batch_size for PPO  (shared hyperparameter) ...")
plot_tune(tune(PPO, {}, "batch_size", bs_vals), "batch_size", "PPO", "tune_ppo_bs.png")

# 3. exploration_fraction (DQN-specific)
print("\n[3/4] Tuning exploration_fraction for DQN  (DQN-specific hyperparameter) ...")
plot_tune(tune(DQN, DQN_BASE, "exploration_fraction", [0.05, 0.1, 0.2, 0.4, 0.6]),
          "exploration_fraction", "DQN", "tune_dqn_ef.png")

# 4. n_steps (PPO-specific)
print("\n[4/4] Tuning n_steps for PPO  (PPO-specific hyperparameter) ...")
plot_tune(tune(PPO, {}, "n_steps", [64, 128, 256, 512, 1024]),
          "n_steps", "PPO", "tune_ppo_nsteps.png")

print("\nPart 2 complete.")
