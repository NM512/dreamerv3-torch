import gym
import numpy as np
import uuid


class CollectDataset:
    def __init__(
        self, env, mode, train_eps, eval_eps=dict(), callbacks=None, precision=32
    ):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None
        self._cache = dict(train=train_eps, eval=eval_eps)[mode]
        self._temp_name = str(uuid.uuid4())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        if isinstance(action, dict):
            transition.update(action)
        else:
            transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        self._episode.append(transition)
        self.add_to_cache(transition)
        if done:
            # detele transitions before whole episode is stored
            del self._cache[self._temp_name]
            self._temp_name = str(uuid.uuid4())
            for key, value in self._episode[1].items():
                if key not in self._episode[0]:
                    self._episode[0][key] = 0 * value
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        # Missing keys will be filled with a zeroed out version of the first
        # transition, because we do not know what action information the agent will
        # pass yet.
        transition["reward"] = 0.0
        transition["discount"] = 1.0
        self._episode = [transition]
        self.add_to_cache(transition)
        return obs

    def add_to_cache(self, transition):
        if self._temp_name not in self._cache:
            self._cache[self._temp_name] = dict()
            for key, val in transition.items():
                self._cache[self._temp_name][key] = [self._convert(val)]
        else:
            for key, val in transition.items():
                if key not in self._cache[self._temp_name]:
                    # fill missing data(action)
                    self._cache[self._temp_name][key] = [self._convert(0 * val)]
                    self._cache[self._temp_name][key].append(self._convert(val))
                else:
                    self._cache[self._temp_name][key].append(self._convert(val))

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        elif np.issubdtype(value.dtype, np.bool):
            dtype = np.bool
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class OneHotAction:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs["reward"] = 0.0
        return obs


class SelectAction:
    def __init__(self, env, key):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return self._env.step(action[self._key])
