from isaaclab.envs import ManagerBasedRLEnv
import torch

class VLAIsaacEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def step(self,action):
        obs, rew, terminated, truncated, info = super().step(action)

        return obs, rew, terminated, truncated, info