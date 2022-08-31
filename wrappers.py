import gym
from gym import spaces
import numpy as np
import torch
import random


# Adapted code from:
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#NoopResetEnv
# Changes: Possible to get also zero NO-OPS + delete the override_num_noops because
class NoopResetEnv(gym.Wrapper):
   def __init__(self, env: gym.Env, maxNoops: int):
      assert env.unwrapped.get_action_meanings()[0] == "NOOP", "Environment has no NOOP"

      gym.Wrapper(env)
      self.__maxNoops = maxNoops
      self.env = env
      

   def reset(self, **kwargs):
      noops = random.randrange(0, self.__maxNoops+1)
      state = self.env.reset(**kwargs)
      for _ in range(noops):
         state, _, done, _ = self.env.step(0)
         if done == True:
            self.env.reset(**kwargs)
      return state


# Adapted from EpisodicLifeEvent-Wrapper from:
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#EpisodicLifeEnv
# Needed for games like Breakout because they got stuck after losing live.
class LiveLostContinueEnv(gym.Wrapper):
   def __init__(self, env:gym.Env):
      gym.Wrapper(env)
      assert env.unwrapped.get_action_meanings()[1] == "FIRE"
      self.env = env
      self.lives = 0

   def step(self, action):
      obs, reward, done, info = self.env.step(action)
      currentLives = self.env.unwrapped.ale.lives()
      if 0 < currentLives < self.lives:
         obs, _, _, _ = self.env.step(1)
      self.lives = currentLives
      return obs, reward, done, info

# Copied from: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#FireResetEnv
class FireResetEnv(gym.Wrapper):

   def __init__(self, env: gym.Env):
      gym.Wrapper.__init__(self, env)
      assert env.unwrapped.get_action_meanings()[1] == "FIRE"
      assert len(env.unwrapped.get_action_meanings()) >= 3

   def reset(self, **kwargs) -> np.ndarray:
      self.env.reset(**kwargs)
      obs, _, done, _ = self.env.step(1)
      if done:
         self.env.reset(**kwargs)
      obs, _, done, _ = self.env.step(2)
      if done:
         self.env.reset(**kwargs)
      return obs


# Reshape the observation to (width, height, channels) that it's compatible for the Conv2D-Layers from PyTorch.
class NumpyWrapper(gym.ObservationWrapper):
   def __init__(self, env, stacked):
      super(NumpyWrapper, self).__init__(env)
      
      self.stacked = stacked
      observation_space = self.observation_space
      
      # stacked frames:
      # observation_space.shape[0]: frames
      # observation_space.shape[1]: width
      # observation_space.shape[2]: height
      # observation_space.shape[3]: channels per frame (exists only when there are more one channel per frame)
      if stacked == True:
         assert len(observation_space.shape) <= 4, "Not implemented for bigger shapes"
         self.k = observation_space.shape[0]
         if len(observation_space.shape) == 4:
            self.k *= observation_space.shape[-1]


         self.w = observation_space.shape[1]
         self.h = observation_space.shape[2]

      # not stacked frames:
      # observation_space.shape[0]: width
      # observation_space.shape[1]: height
      # observation_space.shape[2]: channels (exists only when there are more one channel)
      else:
         assert len(observation_space.shape) <= 3, "Not implemented for bigger shapes"
         self.k = 1
         if len(observation_space.shape) == 3:
            self.k *= observation_space.shape[-1]

         self.w = observation_space.shape[0]
         self.h = observation_space.shape[1]

      self.observation_space = spaces.Box(low=0, high=255, shape=(self.w, self.h, self.k))

   def observation(self, observation):
      observation = np.array(observation)
      if self.stacked == True:      # (nFrames, width, height, channels) -> (width, height, channels, nFrames)
         observation = np.moveaxis(observation, 0, -1)
      observation = np.reshape(observation, (self.w, self.h, self.k))   #Shape: (width, height, channels * nFrames)
      return observation



#Transform observation from numpy to pytorch.
class PyTorchWrapper(gym.ObservationWrapper):
   def __init__(self, env):
      super(PyTorchWrapper, self).__init__(env)
      observation_space = self.observation_space
      self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_space.shape[-1], observation_space.shape[0], observation_space.shape[1]))

   def observation(self, observation):
      observation = np.moveaxis(observation, -1, 0)
      observation = observation / 255.0
      observation = torch.from_numpy(observation).float()
      return observation
