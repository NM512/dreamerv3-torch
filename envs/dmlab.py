import gym
import numpy as np
import deepmind_lab


class DeepMindLabyrinth(object):
    ACTION_SET_DEFAULT = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
        (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
        (0, 0, 0, 0, 1, 0, 0),  # Fire
    )

    ACTION_SET_MEDIUM = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (0, 0, 0, 0, 0, 0, 0),  # Idle.
    )

    ACTION_SET_SMALL = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
    )

    def __init__(
        self,
        level,
        mode,
        action_repeat=4,
        render_size=(64, 64),
        action_set=ACTION_SET_DEFAULT,
        level_cache=None,
        seed=None,
        runfiles_path=None,
    ):
        assert mode in ("train", "test")
        if runfiles_path:
            print("Setting DMLab runfiles path:", runfiles_path)
            deepmind_lab.set_runfiles_path(runfiles_path)
        self._config = {}
        self._config["width"] = render_size[0]
        self._config["height"] = render_size[1]
        self._config["logLevel"] = "WARN"
        if mode == "test":
            self._config["allowHoldOutLevels"] = "true"
            self._config["mixerSeed"] = 0x600D5EED
        self._action_repeat = action_repeat
        self._random = np.random.RandomState(seed)
        self._env = deepmind_lab.Lab(
            level="contributed/dmlab30/" + level,
            observations=["RGB_INTERLEAVED"],
            config={k: str(v) for k, v in self._config.items()},
            level_cache=level_cache,
        )
        self._action_set = action_set
        self._last_image = None
        self._done = True

    @property
    def observation_space(self):
        shape = (self._config["height"], self._config["width"], 3)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({"image": space})

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._action_set))

    def reset(self):
        self._done = False
        self._env.reset(seed=self._random.randint(0, 2**31 - 1))
        obs = self._get_obs()
        return obs

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._action_repeat)
        self._done = not self._env.is_running()
        obs = self._get_obs()
        return obs, reward, self._done, {}

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        del args  # Unused
        del kwargs  # Unused
        return self._last_image

    def close(self):
        self._env.close()

    def _get_obs(self):
        if self._done:
            image = 0 * self._last_image
        else:
            image = self._env.observations()["RGB_INTERLEAVED"]
        self._last_image = image
        return {"image": image}
