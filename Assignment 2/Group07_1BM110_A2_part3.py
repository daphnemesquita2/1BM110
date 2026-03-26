#PART 3: Training the Agent with Action Masking 

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"  

import pickle
import numpy as np
import matplotlib.pyplot as plt
from knapsack_env import BoundedKnapsackEnv                    # requirement 4
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_part3")
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = [9451, 1697, 3229]


TRAIN_TIMESTEPS = 50_000 
TUNE_TIMESTEPS  = 10_000  

def mask_fn(env):
    return env.get_mask()

np.random.seed(SEEDS[0])                                                    
env_inspect = BoundedKnapsackEnv(n_items=200, max_weight=200, mask=True)    
env_inspect = ActionMasker(env_inspect, mask_fn)                            
state_space, _    = env_inspect.reset()                                     
action_space_size = env_inspect.action_space.n                              

print(f"  State space shape : {state_space.shape}")
print(f"  Action space size : {action_space_size}")

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

def make_masked_env():
    """Creates a fresh masked BoundedKnapsackEnv."""
    env = BoundedKnapsackEnv(n_items=200, max_weight=200, mask=True)  
    return ActionMasker(env, mask_fn)                                  
def train(algo_kwargs, seed, total_timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed)                                               
    env   = make_masked_env()
    model = MaskablePPO("MlpPolicy", env, seed=seed, verbose=0,       
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

mppo_rewards = []
mppo_models  = []

print(f"  Total runs: {len(SEEDS)}  (1 algorithm x {len(SEEDS)} seeds)")

for i, seed in enumerate(SEEDS, 1):
    print(f"\n  [{i}/{len(SEEDS)}] Seed={seed}")
    print(f"    MaskablePPO training ...", end=" ", flush=True)
    cb, m = train({}, seed)
    mppo_rewards.append(cb.episode_rewards)
    mppo_models.append(m)
    print(f"done  ({len(cb.episode_rewards)} episodes)")

print("\nAll seeds trained.")

#Plot 

mppo_mean, mppo_std = aggregate(mppo_rewards)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(mppo_mean))
ax.plot(x, mppo_mean, color="seagreen", label="mean")
ax.fill_between(x, mppo_mean - mppo_std, mppo_mean + mppo_std,
                alpha=0.3, color="seagreen", label="±1 std")
ax.set_title("MaskablePPO – Training Reward (3 seeds)")
ax.set_xlabel("Episode (smoothed)")
ax.set_ylabel("Reward (×0.01 scaled)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "part3_training_curves.png"), dpi=150)
print("\n>>> Showing MaskablePPO training curves — close the plot window to continue ...")
plt.show()


mppo_eval_rewards = []
for seed, model in zip(SEEDS, mppo_models):
    np.random.seed(seed)                                               # requirement 2
    eval_env = make_masked_env()
    m, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    mppo_eval_rewards.append(m)

print(f"\nMaskablePPO eval — mean: {np.mean(mppo_eval_rewards):.4f}  "
      f"std: {np.std(mppo_eval_rewards):.4f}  "
      f"best: {np.max(mppo_eval_rewards):.4f}")


def tune(base_kwargs, param_name, values, seeds=SEEDS):
    cache_file = f"cache_tune_MaskablePPO_{param_name}.pkl"

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
            cb, _ = train(kwargs, seed, total_timesteps=TUNE_TIMESTEPS)
            reward_lists.append(cb.episode_rewards)
            print("done")
        results[val] = aggregate(reward_lists)

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    return results

def plot_tune(results, param_name, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    for val, (mean, std) in results.items():
        x = np.arange(len(mean))
        ax.plot(x, mean, label=f"{param_name}={val}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_title(f"MaskablePPO – Tuning {param_name}")
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Reward (×0.01 scaled)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.show()

# n_steps tuning
print("\n[1/1] Tuning n_steps")
ns_results = tune({}, "n_steps", [256, 512, 1024, 2048, 4096])
plot_tune(ns_results, "n_steps", "tune_mppo_nsteps.png")



if os.path.exists(os.path.join(SCRIPT_DIR, "part2_ppo_eval.npy")):
    ppo_eval_rewards = list(np.load(os.path.join(SCRIPT_DIR, "part2_ppo_eval.npy")))
    print(f"\nPPO (Part 2) eval     — mean: {np.mean(ppo_eval_rewards):.4f}  "
          f"std: {np.std(ppo_eval_rewards):.4f}  "
          f"best: {np.max(ppo_eval_rewards):.4f}")
else:
    print("\npart2_ppo_eval.npy not found — run Part 2 first for comparison.")
    ppo_eval_rewards = None

print(f"MaskablePPO eval      — mean: {np.mean(mppo_eval_rewards):.4f}  "
      f"std: {np.std(mppo_eval_rewards):.4f}  "
      f"best: {np.max(mppo_eval_rewards):.4f}")

if ppo_eval_rewards is not None:
    improvement = np.mean(mppo_eval_rewards) - np.mean(ppo_eval_rewards)
    print(f"\nMean improvement from action masking: {improvement:+.4f} "
          f"({'improvement' if improvement > 0 else 'degradation'})")

    # ── Comparison bar chart ──
    _, ax = plt.subplots(figsize=(6, 4))
    labels = ["PPO (Part 2)", "MaskablePPO (Part 3)"]
    means  = [np.mean(ppo_eval_rewards), np.mean(mppo_eval_rewards)]
    stds   = [np.std(ppo_eval_rewards),  np.std(mppo_eval_rewards)]
    colors = ["darkorange", "seagreen"]
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.8)
    ax.set_ylabel("Mean eval reward (×0.01 scaled)")
    ax.set_title("PPO vs MaskablePPO — Evaluation Performance (3 seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "part3_comparison.png"), dpi=150)
    plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([ppo_eval_rewards, mppo_eval_rewards],
                    labels=labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set_color("black")
    ax.set_ylabel("Mean eval reward (×0.01 scaled)")
    ax.set_title("PPO vs MaskablePPO — Evaluation Performance (3 seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "part3_comparison_boxplot.png"), dpi=150)
    plt.show()
