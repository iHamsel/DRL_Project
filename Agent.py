from numbers import Number
import torch
import random
from torch.nn.functional import mse_loss

from ReplayMemory import ReplayMemory
from DecayValue import *
from AgentConfiguration import AgentConfiguration
from TrainingEpisodeLogEntry import TrainingLog

class Agent():

   def __init__(self, config: AgentConfiguration):
      self.config       = config
      self.memory       = ReplayMemory(config.memory_size)
      self.batch_size   = config.learningSize
      self.n_actions    = config.env.action_space.n
      self.eps          = config.epsilon
      self.gamma        = config.gamma
      self.env          = config.env

      self.device       = "cuda" if torch.cuda.is_available() else "cpu"
      self.policy_net   = config.network(config.env.observation_space, config.env.action_space).to(self.device)
      self.target_net   = config.network(config.env.observation_space, config.env.action_space).to(self.device)

      self.updateTargetNetwork()
      self.target_net.eval()
      self.optimizer    = torch.optim.RMSprop(self.policy_net.parameters(), lr=config.learnrate, alpha=0.95, momentum=0.95, eps=0.01)

      self.log = []
      self.trainingRewards    = []
      self.evaluationRewards  = []
      self.training_playSteps = 0
      self.learned = 0
      self.updated = 0

      self.trainLog = TrainingLog()

      self.mode = "random"


   def calcLoss(self, batch) -> torch.Tensor:
      """
         Calculate the loss of a given batch
      """
      states, actions, next_states, rewards = batch
      
      states = torch.stack(states).to(self.device)
      actions  = torch.stack(actions).to(self.device)
      rewards  = torch.stack(rewards).to(self.device)

      non_final_mask    =  torch.BoolTensor( [s is not None for s in next_states])
      non_final_states  =  torch.stack( [s for s in next_states if s is not None]).to(self.device)
      
      curr_Q = self.policy_net(states).gather(1, actions.squeeze(1)).squeeze(1)
      next_Q = self.target_net(non_final_states)
      max_next_Q = next_Q.max(1)[0]

      expected_Q = rewards
      expected_Q[non_final_mask] += self.gamma * max_next_Q

      loss = mse_loss(curr_Q, expected_Q)
      return loss

   def learn(self):
      """
         Learn with experience replay
      """
      if len(self.memory) < self.batch_size:
         return 0
      
      transitions = self.memory.sample(self.batch_size)
      batch = zip(*transitions)

      loss = self.calcLoss(batch)
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
      self.optimizer.step()
      self.eps.step()
      self.learned += 1
      if self.learned % self.config.targetUpdateInterval == 0:
         self.updateTargetNetwork()
         self.updated += 1    
      return float(loss)

   def updateTargetNetwork(self):
      """
         Load the current state of the policy network into the target network
      """
      self.target_net.load_state_dict(self.policy_net.state_dict()) 


   def sample_action(self, state) -> Number:
      """
         Sample a action
      """
      # Explore
      if self.mode == "random" or (self.mode == "train" and random.random() <=self.eps.getValue()):
         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

      state = torch.reshape(state, (1, *state.shape)).to(self.device)
      # Exploit
      with torch.no_grad():
         return self.policy_net(state).max(1)[1].view(1, 1)

   def train(self):
      """
         Train agent
      """
      while self.learned < self.config.trainingLength:
         learnedLeft = self.config.trainingLength - self.learned
         if learnedLeft > self.config.epochLength:
            self.trainEpoch(self.config.epochLength)
         else:
            self.trainEpoch(learnedLeft)

         yield

   def trainEpoch(self, epochLength: int):
      """
         Train agent for 'epochLength' learn-steps
      """
      self.mode = "train"
      self.policy_net.train()

      done = False
      state = self.env.reset()
      losses = []
      episodeReward = 0
      finishedEpisodes = 0

      prevEpisodeSteps  = self.training_playSteps
      prevLearned       = self.learned

      savedParameters = []
      for p in self.policy_net.parameters():
         savedParameters.append(p.data.clone())

      while self.learned - prevLearned < epochLength:
         if done == True:
            avgLoss = sum(losses)/len(losses) if len(losses) > 0 else 0
            episodePlaySteps = self.training_playSteps - prevEpisodeSteps
            prevEpisodeSteps = self.training_playSteps
            self.trainLog.appendEpisode(TrainingLog.EpisodeEntry(episodeReward, episodePlaySteps, avgLoss))
            print(f"Reward for finished episode {self.trainLog.episodesLength()}: {episodeReward}")

            state = self.env.reset()
            losses = []
            episodeReward = 0
            finishedEpisodes += 1
         
         prev_state = state
         state, done, reward, action = self.play(state)

         episodeReward += reward
         if reward > 0:
            reward = 1.0
         elif reward < 0:
            reward = -1.0
         else:
            reward = 0.0
         next_state = state if done == False else None


         self.memory.append(prev_state, action, next_state, reward)
         self.training_playSteps += 1
         if self.training_playSteps % self.config.learnInterval == 0:
            losses.append(self.learn())

      episodes    = self.trainLog.lastEpisodes(finishedEpisodes)
      avgReward   = sum([e.reward for e in episodes]) / len(episodes)
      avgLoss     = sum([e.avgLoss * e.steps / self.config.learnInterval for e in episodes]) + sum(losses)
      avgLoss     = avgLoss / 5000
      
      difference = 0
      n = 0
      for pair in zip(savedParameters, self.policy_net.parameters()):
         res = pair[0] - pair[1].data
         difference += abs(torch.mean(res).detach().item())
         n += 1

      self.trainLog.appendEpoch(TrainingLog.EpochEntry(avgReward, avgLoss, difference/n, finishedEpisodes))
      print(self.trainLog.lastEpoch())
   

   def eval(self, episodes):
      """
         Evaluate agent.
         Agent plays 'episodes' rounds of the game.
         The average of all rewards is saved.
      """
      self.mode = "eval"
      self.policy_net.eval()
      result = 0
      for _ in range(episodes):
         done  = False
         state = self.env.reset()
         episodeReward = 0

         while True:
            state, done, reward, _ = self.play(state)
            episodeReward += reward

            if done == True:
               break
         
         print(f"Evaluation episode reward: {episodeReward}")
         result += episodeReward

      self.evaluationRewards.append(result / episodes)
      print(f"Average reward for evaluation: {result / episodes}")
   
   def fillMemory(self):
      """
         Fills the memory of the agent with predefined number of sequences.
         Agent play game and take always a random action.
      """
      self.mode = "random"
      prefill = self.config.memory_size if self.config.prefill_size > self.config.memory_size else self.config.prefill_size
      state = self.env.reset()
      for _ in range(prefill):
         prev_state = state
         state, done, reward, action = self.play(state)
         next_state = state if done == False else None
         self.memory.append(prev_state, action, next_state, reward)
         if done:
            state = self.env.reset()



   def play(self, currentState):
      """
         Agent "play" one step.
         Agent sample action (random or calculated by network) and repeat it (defined in the config).
      """
      reward = 0
      done = False
      state = None

      action = self.sample_action(currentState)

      for _ in range(self.config.repeatAction):
         state, r, done, _ = self.env.step(action)
         reward += r
         if done:
            break
      return state, done, reward, action

