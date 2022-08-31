from torch import nn
from math import sqrt
import torch

from DQN import DQN

"""
   Backward hook function for rescaling the gradient by factor 1/sqrt(2) of one layer.
"""
def rescale_gradient(m, i, o):
   return tuple([i[0]*(1/sqrt(2))])

"""
   Duel network architecture.
"""
class DuelHead(DQN):
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

      #Forward network for the advantage-stream
      self.adv = nn.Sequential(
         nn.Linear(64*7*7, 512),
         nn.ReLU(),
         nn.Linear(512, out_dim)
      )

      #Forward network for the value-stream
      self.val = nn.Sequential(
         nn.Linear(64*7*7, 512),
         nn.ReLU(),
         nn.Linear(512, 1)
      )
      
      #Rescale the input gradient of both forward streams.
      self.adv.register_full_backward_hook(rescale_gradient)
      self.val.register_full_backward_hook(rescale_gradient)
    
   def forward(self, x: torch.Tensor):
      """
         Args:
               ``x`` (Tensor): Tensor of batched images.
      """
      x = self.cnn(x)
      x = x.view(x.size()[0], -1)
      adv = self.adv(x)
      val = self.val(x)
      x = val + (adv - adv.mean(1, keepdim=True))
      return x