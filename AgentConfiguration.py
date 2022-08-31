from dataclasses import dataclass
from DecayValue import DecayValue, ExponentialDecay
from DQN import DQN
import gym

@dataclass
class AgentConfiguration:
   network:                DQN
   env:                    gym.Env 

   memory_size:            int   = 5000   #Size of the memory
   prefill_size:           int   = 0      #Defines with how many sequences the memory should be prefilled.
   learningSize:           int   = 32     #How many frames are sampled in one learning iteration
   gamma:                  float = 0.95   #Factor how important the long-term reward is
   learnrate:              float = 0.25e-4


   repeatAction:           int   = 4      #Repeat a selected action n times in one play iteration
   learnInterval:          int   = 4      #Learn after n play iterations
   targetUpdateInterval:   int   = 1e4    #Update the target network n learning iterations
   trainingLength:         int   = 1e6    #Determine how long the training should be (in learn steps)
   epochLength:            int   = 5e3    #Determine how long one episode should be (in learn steps)

   epsilon:                DecayValue   = ExponentialDecay(1, 1e-2, 1e6)    #Selecting algorithm for epsilon-greedy