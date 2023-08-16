import gym
import numpy as np


class Crafter:
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        assert task in ("reward", "noreward")
        import crafter

        self._env = crafter.Env(size=size, reward=(task == "reward"), seed=seed)
        self._achievements = crafter.constants.achievements.copy()
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(
                0, 255, self._env.observation_space.shape, dtype=np.uint8
            ),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "log_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        spaces.update(
            {
                f"log_achievement_{k}": gym.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float32
                )
                for k in self._achievements
            }
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        action_space.discrete = True
        return action_space

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        reward = np.float32(reward)
        log_achievements = {
            f"log_achievement_{k}": info["achievements"][k] if info else 0
            for k in self._achievements
        }
        obs = {
            "image": image,
            "is_first": False,
            "is_last": done,
            "is_terminal": info["discount"] == 0,
            "log_reward": np.float32(info["reward"] if info else 0.0),
            **log_achievements,
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self):
        image = self._env.reset()
        obs = {
            "image": image,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
