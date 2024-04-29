import gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from collections import defaultdict
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import json
EnvName = ['Humanoid-v4','Ant-v4', 'Hopper-v4']
tmp_game = 'Humanoid-v4'
model_name = 'TD3'
if model_name == 'TD3':
    model = TD3("MlpPolicy", tmp_game, verbose=1)
elif model_name == 'PPO':
    model = PPO("MlpPolicy", tmp_game, verbose=1)
else:
    model = SAC("MlpPolicy", tmp_game, verbose=1)
print(model_name)
model.learn(total_timesteps=1e6, log_interval=1e3)
model.save(f"{model_name}_{tmp_game}")
'''
# TD3
tmp_game = 'Ant-v4'
env = gym.make(tmp_game)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
tmp_path = f"tmp/sb3_log/TD3/{tmp_game}/"
# set up logger
#new_logger = configure(tmp_path, ["csv", "tensorboard"])
#model.set_logger(new_logger)   
model.learn(total_timesteps=1e6, log_interval=1e3)
model.save(f"TD3_{tmp_game}")
'''

'''
# SAC
tmp_game = 'Humanoid-v4'
tmp_path = f"tmp/sb3_log/sac/{tmp_game}/"
# set up logger
new_logger = configure(tmp_path, ["csv", "tensorboard"])
model = SAC("MlpPolicy", tmp_game, verbose=1)
model.set_logger(new_logger)
    
model.learn(total_timesteps=1e6, log_interval=1e3)
model.save(f"sac_{tmp_game}")
'''

'''
#env = gym.make("Hopper-v4", render_mode="human")
EnvName = ['Humanoid-v4','Ant-v4', 'Hopper-v4']
def get_action_mean():
    for env_name in EnvName:
        # Create the environment
        env = gym.make(env_name, render_mode="human")
        #model = SAC.load(f"SAC_{env_name}")
        model = PPO.load(f"PPO_{env_name}")
        obs, info = env.reset()

        # Other variables
        cnt = 0
        action_sum = None
        reward_sum = 0
        action_average_occurrences = [defaultdict(int) for _ in range(env.action_space.shape[0])]
        while True:
            action, _states = model.predict(obs, deterministic=True)
            if action_sum is None:
                action_sum = np.zeros_like(action)
            #print(action)
            action_sum += action
            cnt += 1
            if cnt % 200 == 0:
                action_average = np.round(action_sum / 200, 3)
                for i, dimension_value in enumerate(action_average):
                    action_average_occurrences[i][dimension_value] += 1
                #print(action_average_occurrences[0])
                action_sum = np.zeros_like(action)
                #print('action_average', action_average)
            if cnt > 1e4:
                action_average_occurrences_json = []
                for occurrences in action_average_occurrences:
                    occurrences_json = {str(k): v for k, v in occurrences.items()}
                    action_average_occurrences_json.append(occurrences_json)
                file_name = f'data/PPO/{env_name}_action_average.json'
                with open(file_name, 'w') as file:
                    json.dump(action_average_occurrences_json, file)
                print(f"Occurrences for {env_name} stored successfully.")
                break
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                print(cnt, reward_sum)
                reward_sum = 0
                obs, info = env.reset()
get_action_mean()
#'''

'''
tmp_game = EnvName[0]
#env = gym.make(tmp_game, render_mode="human")
env = gym.make(tmp_game)
model = SAC.load(f"sac_{tmp_game}")
obs, info = env.reset()
cnt = 0
action_sum = None
while True:
    action, _states = model.predict(obs, deterministic=True)
    if action_sum is None:
        action_sum = np.zeros_like(action)
    #print(action)
    action_sum += action
    cnt += 1
    if cnt % 200 == 0:
        action_average = action_sum / 200
        action_sum = np.zeros_like(action)
        print('action_average', action_average)
        
    if cnt > 1e5:
        break

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(cnt)
        obs, info = env.reset()
'''