from collections import deque
import random
import torch



class ReplayMemory():
   def __init__(self, length):
      assert length > 0, "Parameter 'length' must be > 0"
      self.memory = deque([], maxlen=length)

   def append(self, state: torch.Tensor, action, next_state: torch.Tensor, reward):
      action = action.cpu()
      state = state.cpu()
      if next_state is not None:
         next_state = next_state.cpu()
      reward = torch.tensor(reward, device="cpu")
      self.memory.append((state, action, next_state, reward))

   def sample(self, n):
      assert n > 0, "n must be > 0"
      return random.sample(self.memory, n)

   def __len__(self):
      return len(self.memory)