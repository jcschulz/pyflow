from abc import ABC, abstractmethod
import numpy as np

class ThermodynamicModel(ABC):

    @abstractmethod
    def speed_of_sound(self, density, pressure):
        pass

    @abstractmethod
    def internalenergy(self, density, pressure):
        pass

    @abstractmethod
    def totalenergy(self, density, pressure, u):
        pass

    @abstractmethod
    def pressure(self, density, u, total_energy):
        pass

class EquationOfState:

    R_UNIVERSAL = 8314.46261815324

    def __init__(self, W:float):
        self.R = self.R_UNIVERSAL / W

    def density(self, pressure, temperature):
        return pressure / (self.R * temperature)



class CaloricallyPerfectGas(ThermodynamicModel):
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def speed_of_sound(self, density, pressure):
        return np.sqrt(self.gamma * pressure / density)

    def internalenergy(self, density, pressure):
        return pressure / density / (self.gamma - 1.0)

    def totalenergy(self, density, pressure, u):
        return pressure / (self.gamma - 1) + 0.5 * density * u**2

    def pressure(self, density, u, total_energy):
        return (total_energy - 0.5 * density * u**2 ) * (self.gamma - 1)
