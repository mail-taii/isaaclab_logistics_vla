from .VLAIsaacEnv import VLAIsaacEnv
import torch
class VLA_Evaluator:
    def __init__(self,env_cfg,policy = 'random'):
        self.env = VLAIsaacEnv(cfg=env_cfg)


    def generate_action(self, obs):
        num_envs = self.env.num_envs
        random_action = actions = 2* torch.rand(self.env.action_space.shape, device=self.env.unwrapped.device) - 1
        return random_action
        
    def run_evaluation(self):
        i = 1
        while True:
            with torch.inference_mode():
                actions = self.generate_action(None)
                self.env.step(actions)
                if i%100==0 or i<10:
                    isaac_env = self.env.unwrapped

                    robot_asset = isaac_env.scene.articulations["robot"]
                    
                
                    default_state_tensor = robot_asset.data.root_state_w
                    
                    print("\n" + "="*50)
                    print("Default Root State of 'robot' Asset:")
                    print(f"Shape: {default_state_tensor.shape}")
                    print(f"Data:\n{default_state_tensor[:, 0:3]}")
                    print("="*50 + "\n")
            