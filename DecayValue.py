from math import exp

# super class for  a decay value
class DecayValue():
   def __init__(self, startVal, endVal):
      assert startVal >= 0 and startVal <= 1, "Parameter 'startVal' not in interval [0, 1]"
      assert endVal >= 0 and endVal <= 1, "Parameter 'endVal' not in interval [0, 1]"
      assert endVal <= startVal, "Parameter 'endVal' is bigger than parameter 'startVal'"

      self._startVal = startVal
      self._endVal = endVal
      self._val = self._startVal

   def getValue(self):
      return self._val

   def step(self):
      raise NotImplementedError


# Exponential decay. After n steps the endVal is nearly reached
class ExponentialDecay(DecayValue):
   def __init__(self, startVal, endVal, steps):
      super().__init__(startVal, endVal)
      
      self.__tau = steps / 5     # 5 * Tau = 99%
      self.__t = 0

   def step(self):
      self.__t += 1
      self._val = self._endVal + (self._startVal - self._endVal) * exp(-(self.__t / self.__tau))


# Linear decay. After n steps the endVal is reached.
class LinearDecay(DecayValue):
   def __init__(self, startVal, endVal, steps):
      super().__init__(startVal, endVal)
      self.__steps = steps


   def step(self):
      if self._val > self._endVal:
         self._val -= (self._startVal - self._endVal) / self.__steps
      else:
         self._val = self._endVal


# Every n steps the value is subtracted by the stepsize till endVal is reached.
class StepDecay(DecayValue):
   def __init__(self, startVal, endVal, stepsize, steps):
      super().__init__(startVal, endVal)
      self.__stepsize = stepsize
      self.__steps = steps
      self.__t = 0

   def step(self):
      if self._val <= self._endVal:
         self._val = self._endVal
      else:
         self.__t += 1
         if self.__t == self.__steps:
            self.__t = 0
            self._val -= self.__stepsize

