import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

tmp_path = "tmp/sb3_log/sac/Hopper/"
# set up logger
new_logger = configure(tmp_path, ["csv", "tensorboard"])
model = SAC("MlpPolicy", "Hopper-v4", verbose=1)
model.set_logger(new_logger)
    
model.learn(total_timesteps=1e6, log_interval=100)
model.save("sac_hopper")

#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_pendulum")