import numpy as np
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import copy


class KnapsackEnv(gym.Env):
    '''
    Unbounded Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a
    given weight limit. This version is unbounded meaning that we can select
    items without limit.

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: maximum weight of the knapsack
        3: current weight in knapsack

    Actions:
        Type: Discrete
        0: Place item 0 into knapsack
        1: Place item 1 into knapsack
        2: ...

    Reward:
        Value of item successfully placed into knapsack or 0 if the item
        doesn't fit, at which point the episode ends.

    Starting State:
        Lists of available items and empty knapsack.

    Episode Termination:
        Full knapsack or selection that puts the knapsack over the limit.
    '''

    # Internal list of placed items for better rendering
    _collected_items = []

    def __init__(self, n_items=200, max_weight=200, randomize_params_on_reset=False, mask=False):
        # Generate data with consistent random seed to ensure reproducibility
        self.N = n_items
        self.max_weight = max_weight
        self.current_weight = 0
        self._max_reward = 10000

        self.mask = mask
        self.action_mask = np.ones(self.N, dtype=np.uint8) if self.mask else None

        self.seed = 0

        self.item_numbers = np.arange(self.N)
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)

        self.over_packed_penalty = 0

        self.randomize_params_on_reset = randomize_params_on_reset

        self._collected_items.clear()

        self.set_seed()

        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
            0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)

    def _step(self, item):
        # Check that item will fit
        if self.item_weights[item] + self.current_weight <= self.max_weight:
            self.current_weight += self.item_weights[item]
            reward = self.item_values[item]
            self._collected_items.append(item)
            if self.current_weight == self.max_weight:
                terminated = True
            else:
                terminated = False
        else:
            # End trial if over weight
            reward = self.over_packed_penalty
            terminated = True

        self._update_state()

        return self.state, float(reward), terminated, False, {}    # False is "truncated"

    def _get_obs(self):
        return self.state

    def _update_state(self):
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            self.action_mask = mask

        state = np.vstack([
            self.item_weights,
            self.item_values], dtype=np.int32)
        self.state = np.hstack([
            state,
            np.array([
                [self.max_weight],
                [self.current_weight]])], dtype=np.int32)

    def _reset(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
            self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)
        self.current_weight = 0
        self._collected_items.clear()
        self._update_state()
        self.state = self.state.astype(np.int32)
        return self.state.astype(np.int32)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        return self._reset(), {}

    def step(self, action):
        return self._step(action)


class BoundedKnapsackEnv(KnapsackEnv):
    '''
    Bounded Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a
    given weight limit. This version is bounded meaning each item can be
    selected a limited number of times.

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: list of item limits
        3: maximum weight of the knapsack
        4: current weight in knapsack

    Actions:
        Type: Discrete
        0: Place item 0 into knapsack
        1: Place item 1 into knapsack
        2: ...

    Reward:
        Value of item successfully placed into knapsack or 0 if the item
        doesn't fit, at which point the episode ends.

    Starting State:
        Lists of available items and empty knapsack.

    Episode Termination:
        Full knapsack or selection that puts the knapsack over the limit.
    '''

    def __init__(self, n_items=200, max_weight=200, randomize_params_on_reset=False, mask=False):
        super().__init__(n_items, max_weight, randomize_params_on_reset, mask)
        self.item_limits_init = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        self.item_limits = self.item_limits_init.copy()
        self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
        self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)

        self.observation_space = spaces.Box(0, 1, shape=(3, self.N + 1), dtype=np.float64)

    def get_mask(self):
        return self.action_mask

    def _step(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    terminated = True
                else:
                    terminated = False
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                terminated = True
        else:
            # End if item is unavailable
            reward = 0
            terminated = True

        return self.state / self.max_weight, 0.01 * float(reward), terminated, False, {}    # False is 'truncated'

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ], dtype=np.int32)
        state = np.hstack([
            state_items,
            np.array([[self.max_weight],
                      [self.current_weight],
                      [0]  # Serves as place holder
                      ], dtype=np.int32)
        ])

        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            self.action_mask = np.where(self.item_limits > 0, mask, 0)

        self.state = state.copy()

    def _reset(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
            self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)
            self.item_limits = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        else:
            self.item_limits = self.item_limits_init.copy()

        self.current_weight = 0
        self._update_state()


        return self.state / self.max_weight
