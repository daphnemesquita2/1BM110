##########################################################

#### PART 3: Training the Agent with Action Masking ######

##########################################################

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress TensorFlow oneDNN info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"  # suppress all TensorFlow logging

import pickle
import numpy as np
import matplotlib.pyplot as plt
from knapsack_env import BoundedKnapsackEnv                    # requirement 4
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ──────────────────────────────────────────────
# Seeds — load from Part 2 (requirement: same
# 3 seeds as Part 2)
# ──────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_part3")
os.makedirs(FIGURES_DIR, exist_ok=True)
SEEDS_FILE  = os.path.join(SCRIPT_DIR, "part2_seeds.npy")

if os.path.exists(SEEDS_FILE):
    SEEDS = [int(x) for x in np.load(SEEDS_FILE)]
    print(f"Loaded seeds from Part 2: {SEEDS}")
else:
    raise FileNotFoundError(
        "part2_seeds.npy not found — run Part 2 first so the same seeds are used."
    )


# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────

TRAIN_TIMESTEPS = 50_000   # required by assignment (same as Part 2)
TUNE_TIMESTEPS  = 10_000   # reduced for tuning — relative comparison only


# ──────────────────────────────────────────────
# mask_fn: returns the action mask from the env
# (requirement 5b — implementation left to us)
# 1 = action is valid, 0 = action is invalid
# ──────────────────────────────────────────────

def mask_fn(env):
    return env.get_mask()


# ──────────────────────────────────────────────
# Requirement 6: Inspect environment
# ──────────────────────────────────────────────

np.random.seed(SEEDS[0])                                                    # requirement 2
env_inspect = BoundedKnapsackEnv(n_items=200, max_weight=200, mask=True)    # requirement 5a
env_inspect = ActionMasker(env_inspect, mask_fn)                            # requirement 5b
state_space, _    = env_inspect.reset()                                     # requirement 6
action_space_size = env_inspect.action_space.n                              # requirement 6

print("\n========================================")
print("Environment Inspection (masked)")
print("========================================")
print(f"  State space shape : {state_space.shape}")
print(f"  Action space size : {action_space_size}")
print(f"  (3 rows: weights, values, limits  |  201 cols: 200 items + 1 knapsack info)")
print(f"  Action masking    : enabled — invalid actions set to -inf before softmax")


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

def make_masked_env():
    """Creates a fresh masked BoundedKnapsackEnv."""
    env = BoundedKnapsackEnv(n_items=200, max_weight=200, mask=True)  # requirement 5a
    return ActionMasker(env, mask_fn)                                  # requirement 5b

def train(algo_kwargs, seed, total_timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed)                                               # requirement 2
    env   = make_masked_env()
    model = MaskablePPO("MlpPolicy", env, seed=seed, verbose=0,       # requirement 3
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
# PART A: Train MaskablePPO — 3 seeds, 50k steps
# Using default PPO hyperparameters (same as Part 2)
# ──────────────────────────────────────────────

mppo_rewards = []
mppo_models  = []

print("\n========================================")
print("PART A: Training MaskablePPO (3 seeds, 50k steps each)")
print(f"  Total runs: {len(SEEDS)}  (1 algorithm x {len(SEEDS)} seeds)")
print(f"  Hyperparameters: PPO defaults (same as Part 2)")
print("========================================")

for i, seed in enumerate(SEEDS, 1):
    print(f"\n  [{i}/{len(SEEDS)}] Seed={seed}")
    print(f"    MaskablePPO training ...", end=" ", flush=True)
    cb, m = train({}, seed)
    mppo_rewards.append(cb.episode_rewards)
    mppo_models.append(m)
    print(f"done  ({len(cb.episode_rewards)} episodes)")

print("\nAll seeds trained.")


# ── Plot training curves (aggregated mean ± std) ──

print("\n----------------------------------------")
print("Plotting MaskablePPO training curves (mean ± std across 3 seeds) ...")
print("----------------------------------------")

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


# ── Evaluate MaskablePPO (all 3 seeds) ──

print("\n----------------------------------------")
print("Evaluating MaskablePPO (10 episodes per seed, all 3 seeds) ...")
print("----------------------------------------")

mppo_eval_rewards = []
for seed, model in zip(SEEDS, mppo_models):
    np.random.seed(seed)                                               # requirement 2
    eval_env = make_masked_env()
    m, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    mppo_eval_rewards.append(m)

print(f"\nMaskablePPO eval — mean: {np.mean(mppo_eval_rewards):.4f}  "
      f"std: {np.std(mppo_eval_rewards):.4f}  "
      f"best: {np.max(mppo_eval_rewards):.4f}")
print(f"(Reminder: multiply by 100 for real reward — e.g. 0.93 → 93)")


# ──────────────────────────────────────────────
# PART B: Hyperparameter Tuning
# Start from PPO defaults, explore new n_steps
# values to see if masking improves further.
# At least 3 new values required by assignment.
# ──────────────────────────────────────────────

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
    print(f"  Saved to cache: {cache_file}")
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
    print(f"\n>>> Showing MaskablePPO – {param_name} tuning plot — close the plot window to continue ...")
    plt.show()


print("\n========================================")
print("PART B: Hyperparameter Tuning")
print("  Tuning n_steps — at least 3 new values beyond Part 2 PPO tuning")
print(f"  Using same seeds: {SEEDS}")
print("========================================")

# n_steps tuning — new values beyond those tested in Part 2 PPO
print("\n[1/1] Tuning n_steps for MaskablePPO ...")
ns_results = tune({}, "n_steps", [256, 512, 1024, 2048, 4096])
plot_tune(ns_results, "n_steps", "tune_mppo_nsteps.png")


# ──────────────────────────────────────────────
# PART C: Compare MaskablePPO vs PPO (Part 2)
# ──────────────────────────────────────────────

print("\n========================================")
print("PART C: Comparison — MaskablePPO vs PPO (Part 2)")
print("========================================")

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
    print("\n>>> Showing comparison bar chart — close the plot window to continue ...")
    plt.show()

print("\nPart 3 complete.")
