from torch import isin, nn
from gym import Space


"""
   Super class for the DQNs (SingleHead, DuelHead)
"""
class DQN(nn.Module):
   def __init__(self, observation_space, action_space):
      super().__init__()

      assert isinstance(observation_space, Space), "Parameter 'observation_space' is no instance of 'Space'"
      assert isinstance(action_space, Space), "Paramter 'action_space' is no instance of 'Space'"

      self.observation_space = observation_space
      self.action_space = action_space

   def forward(self, x):
      raise NotImplementedError





