import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):
    def __init__(self, battery_capacity=10, max_charge_rate=5, max_discharge_rate=5, grid_price=2, solar_price=0.1):
        super(BatteryEnv, self).__init__()

        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.grid_price = grid_price
        self.solar_price = solar_price

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        self.reset()

    def step(self, action):
        reward = 0
        terminated = False

        energy_from_solar = self.solar_generation
        energy_demand = self.energy_demand
        
        if action == 0:
            charge_amount = min(energy_from_solar - energy_demand, self.max_charge_rate, self.battery_capacity - self.battery_level)
            if charge_amount > 0:
                self.battery_level += charge_amount
                reward += charge_amount * self.solar_price
        
        elif action == 1:
            discharge_amount = min(energy_demand - energy_from_solar, self.max_discharge_rate, self.battery_level)
            if discharge_amount > 0:
                self.battery_level -= discharge_amount
                energy_demand -= discharge_amount
        
        net_energy = energy_from_solar - energy_demand
        if net_energy < 0:
            reward -= abs(net_energy) * self.grid_price

        self.solar_generation = np.random.uniform(0, 10)
        self.energy_demand = np.random.uniform(0, 10)

        return self._get_obs(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery_level = np.random.uniform(0, self.battery_capacity)
        
        self.solar_generation = np.random.uniform(0, 10)
        self.energy_demand = np.random.uniform(0, 10)

        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        print(f"Battery Level: {self.battery_level:.2f} kWh, Solar Generation: {self.solar_generation:.2f} kWh, Energy Demand: {self.energy_demand:.2f} kWh")

    def _get_obs(self):
        return np.array([self.battery_level, self.solar_generation, self.energy_demand], dtype=np.float32)
