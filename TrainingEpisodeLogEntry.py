from dataclasses import dataclass

# log for one game episode
@dataclass
class TrainingEpisodeLogEntry:
   reward:        float
   avgLoss:       float
   playSteps:     int
   learnSteps:    int
   targetUpdates: int


   def __str__(self) -> str:
      return f"""
         ########################\n
         Reward: {self.reward}
         Average loss: {self.avgLoss}
         # """



# Training log. 
class TrainingLog:

   #Entry for one game episode.
   @dataclass
   class EpisodeEntry:
      reward:  float
      steps:   int
      avgLoss: float
      
      def __str__(self) -> str:
         return f"""
            ########################
            Reward: {self.reward}
            Played steps: {self.steps}
            Average loss: {self.avgLoss}
            ########################
         """


   #Entry for one training epoch.
   @dataclass
   class EpochEntry:
      avgReward:        float
      avgLoss:          float
      weightDifference: float
      finishedEpisodes: int

      def __str__(self) -> str:
         return f"""
            ########################
            Average reward: {self.avgReward}
            Average loss: {self.avgLoss}
            Average weight difference: {self.weightDifference}
            Finished episodes: {self.finishedEpisodes}
            ########################
         """


   
   def __init__(self) -> None:
      self.__episodes  = []
      self.__epochs    = []

   def appendEpisode(self, episode: EpisodeEntry) -> None:
      self.__episodes.append(episode)

   def appendEpoch(self, epoch: EpochEntry) -> None:
      self.__epochs.append(epoch)

   def lastEpisode(self) -> EpisodeEntry:
      return self.__episodes[-1]

   def lastEpisodes(self, n: int) -> list:
      return self.__episodes[-n:]

   def episodes(self) -> list:
      return self.__episodes

   def episodesLength(self) -> int:
      return len(self.__episodes)

   def lastEpoch(self) -> EpochEntry:
      return self.__epochs[-1]

   def epochs(self) -> list:
      return self.__epochs

   def epochsLength(self) -> int:
      return len(self.__epochs)

   

   

   
