import os
from battery_env import BatteryEnv
from stable_baselines3 import PPO

script_dir = os.path.dirname(__file__)
model_save_path = os.path.join(script_dir, "ppo_battery_manager")

env = BatteryEnv()

model = PPO("MlpPolicy", env, verbose=1)

print("--- Starting RL Agent Training ---")
model.learn(total_timesteps=10000)
print("--- RL Agent Training Finished ---")

model.save(model_save_path)
print(f"Agent saved successfully to '{model_save_path}.zip'")

print("\n--- Testing the trained agent ---")
obs, _ = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    print(f"Step {i+1}:")
    env.render()
    print(f"Action taken: {['Charge', 'Discharge', 'Idle'][action]}, Reward received: {rewards:.2f}\n")
