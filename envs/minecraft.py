import numpy as np
from . import minecraft_base

import gym


def make_env(task, *args, **kwargs):
    return {
        "wood": MinecraftWood,
        "climb": MinecraftClimb,
        "diamond": MinecraftDiamond,
    }[task](*args, **kwargs)


class MinecraftWood:
    def __init__(self, *args, **kwargs):
        actions = BASIC_ACTIONS
        self.rewards = [
            CollectReward("log", repeated=1),
            HealthReward(),
        ]
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        return obs, reward, done, info


class MinecraftClimb:
    def __init__(self, *args, **kwargs):
        actions = BASIC_ACTIONS
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
        self._previous = None
        self._health_reward = HealthReward()
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x, y, z = obs["log_player_pos"]
        height = np.float32(y)
        if obs["is_first"]:
            self._previous = height
        reward = height - self._previous
        reward += self._health_reward(obs)
        self._previous = height
        return obs, reward, done, info


class MinecraftDiamond(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        actions = {
            **BASIC_ACTIONS,
            "craft_planks": dict(craft="planks"),
            "craft_stick": dict(craft="stick"),
            "craft_crafting_table": dict(craft="crafting_table"),
            "place_crafting_table": dict(place="crafting_table"),
            "craft_wooden_pickaxe": dict(nearbyCraft="wooden_pickaxe"),
            "craft_stone_pickaxe": dict(nearbyCraft="stone_pickaxe"),
            "craft_iron_pickaxe": dict(nearbyCraft="iron_pickaxe"),
            "equip_stone_pickaxe": dict(equip="stone_pickaxe"),
            "equip_wooden_pickaxe": dict(equip="wooden_pickaxe"),
            "equip_iron_pickaxe": dict(equip="iron_pickaxe"),
            "craft_furnace": dict(nearbyCraft="furnace"),
            "place_furnace": dict(place="furnace"),
            "smelt_iron_ingot": dict(nearbySmelt="iron_ingot"),
        }
        self.items = [
            "log",
            "planks",
            "stick",
            "crafting_table",
            "wooden_pickaxe",
            "cobblestone",
            "stone_pickaxe",
            "iron_ore",
            "furnace",
            "iron_ingot",
            "iron_pickaxe",
            "diamond",
        ]
        self.rewards = [CollectReward(item, once=1) for item in self.items] + [
            HealthReward()
        ]
        env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        # restrict log for memory save
        obs = {
            k: v
            for k, v in obs.items()
            if "log" not in k or k.split("/")[-1] in self.items
        }
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # called for reset of reward calculations
        _ = sum([fn(obs, self.env.inventory) for fn in self.rewards])
        # restrict log for memory save
        obs = {
            k: v
            for k, v in obs.items()
            if "log" not in k or k.split("/")[-1] in self.items
        }
        return obs


class CollectReward:
    def __init__(self, item, once=0, repeated=0):
        self.item = item
        self.once = once
        self.repeated = repeated
        self.previous = 0
        self.maximum = 0

    def __call__(self, obs, inventory):
        current = inventory[self.item]
        if obs["is_first"]:
            self.previous = current
            self.maximum = current
            return 0
        reward = self.repeated * max(0, current - self.previous)
        if self.maximum == 0 and current > 0:
            reward += self.once
        self.previous = current
        self.maximum = max(self.maximum, current)
        return reward


class HealthReward:
    def __init__(self, scale=0.01):
        self.scale = scale
        self.previous = None

    def __call__(self, obs, inventory=None):
        health = obs["health"]
        if obs["is_first"]:
            self.previous = health
            return 0
        reward = self.scale * (health - self.previous)
        self.previous = health
        return sum(reward)


BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=1),
    "turn_up": dict(camera=(-15, 0)),
    "turn_down": dict(camera=(15, 0)),
    "turn_left": dict(camera=(0, -15)),
    "turn_right": dict(camera=(0, 15)),
    "forward": dict(forward=1),
    "back": dict(back=1),
    "left": dict(left=1),
    "right": dict(right=1),
    "jump": dict(jump=1, forward=1),
    "place_dirt": dict(place="dirt"),
}
