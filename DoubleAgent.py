import torch
from torch.nn.functional import mse_loss
from Agent import Agent
from AgentConfiguration import AgentConfiguration







"""
   Implementation of the DDQN algorithmn.
   Only needed to change the calcLoss-method.
"""
class DoubleAgent(Agent):
   def __init__(self, config: AgentConfiguration):
      super().__init__(config)


   def calcLoss(self, batch):
      states, actions, next_states, rewards = batch
      
      states = torch.stack(states).to(self.device)
      actions  = torch.stack(actions).to(self.device)
      rewards  = torch.stack(rewards).to(self.device)

      non_final_mask    =  torch.BoolTensor( [s is not None for s in next_states])
      non_final_states  =  torch.stack( [s for s in next_states if s is not None]).to(self.device)


      curr_Q = self.policy_net(states).gather(1, actions.squeeze(1)).squeeze(1)
      
      #calculate which action in next step should be used.
      next_Q = self.policy_net(non_final_states)
      _, next_actions = next_Q.max(1)

      #calculate q-values for the next step by the target network. Choose these q-values which corresponds to the "next_actions".
      target_next_Q = self.target_net(non_final_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

      expected_Q = rewards
      expected_Q[non_final_mask] += self.gamma * target_next_Q
      loss = mse_loss(curr_Q, expected_Q)
      return loss



