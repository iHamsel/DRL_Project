from torch import nn
import torch

from DQN import DQN

"""
   Normal network architecture.
"""
class SingleHead(DQN):
   def __init__(self, observation_space, action_space):
      super().__init__(observation_space, action_space)

      in_dim   = observation_space.shape[0]
      out_dim  = action_space.n
      self.cnn = nn.Sequential(
         nn.Conv2d(in_dim, 32, kernel_size=8, stride=4),
         nn.ReLU(),
         nn.Conv2d(32, 64, kernel_size=4, stride=2),
         nn.ReLU(),
         nn.Conv2d(64, 64, kernel_size=3),
         nn.ReLU()
      )

      #Fully connected network for predicting q values for each action
      self.fc = nn.Sequential(
         nn.Linear(64*7*7, 1024),
         nn.ReLU(),
         nn.Linear(1024, out_dim)
      )

    
   def forward(self, x: torch.Tensor):
      """
         Args:
               ``x`` (Tensor): Tensor of batched images.
      """
      x = self.cnn(x)
      x = x.view(x.size()[0], -1)
      x = self.fc(x)
      return x