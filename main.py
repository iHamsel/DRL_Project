from Agent import Agent
from AgentConfiguration import AgentConfiguration
from DoubleAgent import DoubleAgent
from DecayValue import LinearDecay

import gym
import wrappers
from DQN import SingleHead, DuelHead
import pickle
import torch



gamename = "IceHockey"     #Game name which should be played.

env = gym.make(f"{gamename}NoFrameskip-v4")
env = gym.wrappers.ResizeObservation(env, (84, 84))   # Resize all observations to 84x84
env = gym.wrappers.GrayScaleObservation(env)          # Use grayscale observation like in the paper.
env = gym.wrappers.FrameStack(env, 4)                 # Stack four frames to one sequence.
env = wrappers.NumpyWrapper(env, True)                # Reshape observation into numpy-array with a shape which can easily used by PyTorch
env = wrappers.PyTorchWrapper(env)                    # Numpy -> PyTorch
env = wrappers.NoopResetEnv(env, 30)                  # Do noops on reset (to start from random positions)
env = wrappers.LiveLostContinueEnv(env)               # If agent lose live game should be continued
env = wrappers.FireResetEnv(env)                      # In some environments you need to fire when game is reseted


"""
   Configuration for normal network architecture
"""
singleConfig = AgentConfiguration(
   network=SingleHead,
   env=env,
   memory_size=int(5e4),
   prefill_size=int(2.5e4),
   learningSize=32,
   gamma=0.99,
   learnrate=2e-4,
   epsilon=LinearDecay(1, 0.1, 1e5),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=500,
   trainingLength=1e6,
   epochLength=5e3
   )

"""
   Configuration for duel network architecture (vary only in the used network)
"""
duelConfig = AgentConfiguration(
   network=DuelHead,
   env=env,
   memory_size=int(5e3),
   prefill_size=int(2.5e3),
   learningSize=32,
   gamma=0.99,
   learnrate=2e-4,
   epsilon=LinearDecay(1, 0.1, 1e5),
   repeatAction=4,
   learnInterval=4,
   targetUpdateInterval=500,
   trainingLength=1e6,
   epochLength=5e3
   )


combinations = [
   {"agent": Agent(singleConfig), "name": "SingleDQN"},
   {"agent": DoubleAgent(singleConfig), "name": "SingleDDQN"},
   {"agent": Agent(duelConfig), "name": "DuelDQN"},
   {"agent": DoubleAgent(duelConfig), "name": "DuelDDQN"},
]


for combination in combinations:
   agentname = combination["name"]
   agent: Agent = combination["agent"]
   prefix = f"{gamename}_{agentname}"
   agent.fillMemory()

   #train model (agent.train() is a generator)
   for _ in agent.train():

      #save training results
      savename = f"{prefix}_training.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.trainLog, f)
      
      #save trained model
      torch.save(agent.policy_net.state_dict(), f"{prefix}.model")

      #print informations about training epoch
      print(f"Played: {agent.training_playSteps} | Learned: {agent.learned} | Target updated: {agent.updated} | Epsilon value: {agent.eps.getValue()}")
      
      #do evaluation
      agent.eval(10)

      #save evaluation results
      savename = f"{prefix}_evaluation.pickle"
      with open(f"{savename}", "wb") as f:
         pickle.dump(agent.evaluationRewards, f)

   #free resources (replay memory, ...)
   del agent   

