from abc import ABC, abstractmethod


class AbstractMPCC(ABC):
    def __init__(self, pi):
        pass

    @abstractmethod
    def getvars(self):
        pass

    @abstractmethod
    def complementarity(self, initial_guess):
        pass

    @abstractmethod
    def objective(self, constraints):
        pass

    @abstractmethod
    def gradient(self, objective):
        pass

    @abstractmethod
    def constraints(self, variables):
        pass

    @abstractmethod
    def jacobian(self, constraints):
        pass
