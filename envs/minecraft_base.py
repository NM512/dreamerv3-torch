import logging
import threading

import numpy as np
import gym


class MinecraftBase(gym.Env):
    _LOCK = threading.Lock()

    def __init__(
        self,
        actions,
        repeat=1,
        size=(64, 64),
        break_speed=100.0,
        gamma=10.0,
        sticky_attack=30,
        sticky_jump=10,
        pitch_limit=(-60, 60),
        logs=False,
    ):
        if logs:
            logging.basicConfig(level=logging.DEBUG)
        self._repeat = repeat
        self._size = size
        if break_speed != 1.0:
            sticky_attack = 0

        # Make env
        with self._LOCK:
            from . import minecraft_minerl

            self._env = minecraft_minerl.MineRLEnv(size, break_speed, gamma).make()
        self._inventory = {}

        # Observations
        self._inv_keys = [
            k
            for k in self._flatten(self._env.observation_space.spaces)
            if k.startswith("inventory/")
            if k != "inventory/log2"
        ]
        self._max_inventory = None
        self._equip_enum = self._env.observation_space["equipped_items"]["mainhand"][
            "type"
        ].values.tolist()

        # Actions
        self._noop_action = minecraft_minerl.NOOP_ACTION
        actions = self._insert_defaults(actions)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f"Minecraft action space ({len(self._action_values)}):"
        print(message, ", ".join(self._action_names))
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
                "inventory": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._inv_keys),), dtype=np.float32
                ),
                "inventory_max": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._inv_keys),), dtype=np.float32
                ),
                "equipped": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._equip_enum),), dtype=np.float32
                ),
                "health": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "hunger": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "breath": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                **{
                    f"log_{k}": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int64)
                    for k in self._inv_keys
                },
                "log_player_pos": gym.spaces.Box(
                    -np.inf, np.inf, (3,), dtype=np.float32
                ),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.discrete.Discrete(len(self._action_values))
        space.discrete = True
        return space

    def step(self, action):
        action = action.copy()
        action = self._action_values[action]
        action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self._env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", done))

        obs = self._obs(obs)
        assert "pov" not in obs, list(obs.keys())
        return obs, reward, done, info

    @property
    def inventory(self):
        return self._inventory

    def reset(self):
        # inventory will be added in _obs
        self._inventory = {}
        self._max_inventory = None

        with self._LOCK:
            obs = self._env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def _obs(self, obs):
        obs = self._flatten(obs)
        obs["inventory/log"] += obs.pop("inventory/log2")
        self._inventory = {
            k.split("/", 1)[1]: obs[k] for k in self._inv_keys if k != "inventory/air"
        }
        inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
        if self._max_inventory is None:
            self._max_inventory = inventory
        else:
            self._max_inventory = np.maximum(self._max_inventory, inventory)
        index = self._equip_enum.index(obs["equipped_items/mainhand/type"])
        equipped = np.zeros(len(self._equip_enum), np.float32)
        equipped[index] = 1.0
        player_x = obs["location_stats/xpos"]
        player_y = obs["location_stats/ypos"]
        player_z = obs["location_stats/zpos"]
        obs = {
            "image": obs["pov"],
            "inventory": inventory,
            "inventory_max": self._max_inventory.copy(),
            "equipped": equipped,
            "health": np.float32([obs["life_stats/life"]]) / 20,
            "hunger": np.float32([obs["life_stats/food"]]) / 20,
            "breath": np.float32([obs["life_stats/air"]]) / 300,
            "is_first": obs["is_first"],
            "is_last": obs["is_last"],
            "is_terminal": obs["is_terminal"],
            **{f"log_{k}": np.int64(obs[k]) for k in self._inv_keys},
            "log_player_pos": np.array([player_x, player_y, player_z], np.float32),
        }
        for key, value in obs.items():
            space = self.observation_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            assert (key, value, value.dtype, value.shape, space)
        return obs

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = 1
                action["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = 1
                action["forward"] = 1
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action

    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result
