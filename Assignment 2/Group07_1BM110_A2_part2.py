##########################################

#### PART 2: Training the Agent ##########

##########################################


import numpy as np
from knapsack_env import BoundedKnapsackEnv
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


'''Set up using AI'''

# Set seed
seed = 2024
np.random.seed(seed)

# Create environment
env = BoundedKnapsackEnv(n_items=200, max_weight=200)

# For Part 3 (masked version):
def mask_fn(env):
    return env.get_mask()

env = BoundedKnapsackEnv(n_items=200, max_weight=200, mask=True)
env = ActionMasker(env, mask_fn)

# Create model (seed set here too)
model = DQN("MlpPolicy", env, seed=seed)