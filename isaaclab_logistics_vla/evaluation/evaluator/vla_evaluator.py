from .VLAIsaacEnv import VLAIsaacEnv
import os
import torch
import numpy as np
import time


class VLA_Evaluator:
    def __init__(self, env_cfg, policy='random', from_json=2):
        self.from_json = from_json
        
        #---在初始化环境之前，将参数注入配置---
        if hasattr(env_cfg.commands, "object_commands"):
            env_cfg.commands.object_commands.from_json = self.from_json
            print(f"[Evaluator]指令模式已设置为: {self.from_json}(0:录制, 1:回放, 2:随机)")

        # 初始化环境
        self.env = VLAIsaacEnv(cfg=env_cfg)
        self.isprint = False

        self.lift_duration = 250  
        self.step_counter = 0    
        
        # ... 保持原有的 RRT 路径加载逻辑不变 ...
        txt_path = '/home/wst/code/ompl/RRT_path.txt'
        if os.path.exists(txt_path):
            self.action_trajectory = self._load_and_process_txt(txt_path)
            print(f"[INFO] Successfully loaded {len(self.action_trajectory)} steps from {txt_path}")
        else:
            self.action_trajectory = None

    def _load_and_process_txt(self, file_path):
        """
        读取带 [ ] 和 , 的 TXT，并将顺序格式转换为交错格式
        """
        raw_data_list = []
        
        # === 1. 手动读取并清洗数据 ===
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue

                clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
                

                row_values = np.fromstring(clean_line, sep=" ")
                
                if len(row_values) > 0:
                    raw_data_list.append(row_values)
        raw_data = np.array(raw_data_list)

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
            
        T, D = raw_data.shape
        if D != 14:
            raise ValueError(f"TXT data must have 14 columns, but got {D}")

        left_arm = raw_data[:, 0:7]   # 前7个
        right_arm = raw_data[:, 7:14] # 后7个
        
        interleaved_data = np.zeros_like(raw_data)
        interleaved_data[:, 0::2] = left_arm  # 偶数位放左臂
        interleaved_data[:, 1::2] = right_arm # 奇数位放右臂
        
        return torch.tensor(interleaved_data, dtype=torch.float32, device=self.env.device)


    def generate_action(self, obs):
        actions = torch.zeros((self.env.num_envs, 17), device=self.env.device)

        actions[:, 16] = 0.5 
        actions[:, 14] = 0.0
        actions[:, 15] = 0.0

        if self.step_counter < self.lift_duration:
            pass 
            
        else:
            if self.action_trajectory is not None:
                traj_idx = self.step_counter - self.lift_duration
                
                if traj_idx < len(self.action_trajectory):
                    current_pose = self.action_trajectory[traj_idx]
                    
                    actions[:, 0:14] = current_pose 
                else:
                    # 如果数据读完了，通常策略是：保持最后一帧
                    actions[:, 0:14] = self.action_trajectory[-1]
        
        self.step_counter += 1
        
        return actions
        
    def run_evaluation(self):
        i = 1
        # 环境 reset 时会根据 from_json 决定是随机生成、记录 JSON 还是读取 JSON
        self.env.reset()

        print(f"[INFO] Evaluation started. Mode: {self.from_json}")

        while True:
            with torch.inference_mode():
                actions = self.generate_action(None)
                obs, rew, terminated, truncated, info = self.env.step(actions)
                #Atime.sleep(1)
                if i%100==0 or i<10:
                    isaac_env = self.env.unwrapped

                    robot_asset = isaac_env.scene.articulations["robot"]
                    
                
                    default_state_tensor = robot_asset.data.root_state_w
                    
                    print("\n" + "="*50)
                    print("Default Root State of 'robot' Asset:")
                    print(f"Shape: {default_state_tensor.shape}")
                    print(f"Data:\n{default_state_tensor[:, 0:3]}")
                    print(f"Reward :\n{rew}")
                    print("="*50 + "\n")
            