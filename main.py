import os
import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from rl.battery_env import BatteryEnv

def main():
    print("--- Starting Smart Battery Management Simulation ---")

    script_dir = os.path.dirname(__file__)
    ml_model_path = os.path.join(script_dir, 'ml', 'solar_power_prediction_model.joblib')
    rl_model_path = os.path.join(script_dir, 'rl', 'ppo_battery_manager.zip')

    print("Loading trained models...")
    try:
        ml_model = joblib.load(ml_model_path)
        rl_model = PPO.load(rl_model_path)
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"\nERROR: Model file not found: {e.filename}")
        print("Please make sure you have trained the models first by running:")
        print("1. python src/train_ml_model.py")
        print("2. python src/rl/train_agent.py")
        return

    sim_env = BatteryEnv(battery_capacity=10, max_charge_rate=5, max_discharge_rate=5)
    obs, _ = sim_env.reset()
    total_reward = 0
    num_steps = 24

    print(f"\n--- Running simulation for {num_steps} steps ---")

    feature_names = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'day_of_year']

    for step in range(num_steps):
        hour = step
        if 6 <= hour <= 18:
            irradiation = np.random.uniform(0.3, 1.0)
            ambient_temp = np.random.uniform(18, 35)
            module_temp = np.random.uniform(25, 60)
        else:
            irradiation = 0.0
            ambient_temp = np.random.uniform(10, 20)
            module_temp = ambient_temp
        
        day_of_year = 150
        energy_demand = np.random.uniform(0.5, 3.0)

        ml_input_df = pd.DataFrame([[ambient_temp, module_temp, irradiation, hour, day_of_year]], columns=feature_names)
        
        predicted_solar_watts = ml_model.predict(ml_input_df)[0]
        predicted_solar_kwh = max(0, predicted_solar_watts / 1000)

        sim_env.solar_generation = predicted_solar_kwh
        sim_env.energy_demand = energy_demand
        current_obs = sim_env._get_obs()

        action, _ = rl_model.predict(current_obs, deterministic=True)

        battery_state_before_action = sim_env.battery_level
        obs, reward, terminated, _, info = sim_env.step(action)
        total_reward += reward

        print(f"\n--- Step {step + 1} (Hour {hour}:00) ---")
        print(f"Conditions: Solar Prediction={predicted_solar_kwh:.2f} kWh, Demand={energy_demand:.2f} kWh")
        print(f"State before action: Battery={battery_state_before_action:.2f} kWh")
        print(f"Decision: RL Agent chose to --> {['CHARGE', 'DISCHARGE', 'IDLE'][action]}")
        sim_env.render()
        print(f"Result: Reward for this step = {reward:.2f}")

    print("\n--- Simulation Finished ---")
    print(f"Total reward over {num_steps} hours: {total_reward:.2f}")

if __name__ == "__main__":
    main()
