######################################################################

#### PART 3: Training the Agent with Invalid Action Masking###########

######################################################################


import numpy as np
from knapsack_env import BoundedKnapsackEnv
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker



