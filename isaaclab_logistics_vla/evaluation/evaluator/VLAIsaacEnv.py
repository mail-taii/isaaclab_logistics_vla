from isaaclab.envs import ManagerBasedRLEnv

class VLAIsaacEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_entity = self.scene["robot"]
        self.joint_index_dict = self._build_joint_mapping()
        print(self.joint_index_dict)
    
    def step(self,action):
        obs, rew, terminated, truncated, info = super().step(action)

        return obs, rew, terminated, truncated, info
    
    def _build_joint_mapping(self):
        """
        获取关节-索引字典，其中夹爪视为一个关节
        """
        mapping = {}
        current_idx = 0 
        
        action_man = self.action_manager
        robot = self.scene["robot"]
        all_joint_names = robot.data.joint_names # 机器人所有关节的名称列表
        
        term_names = action_man.active_terms
        if isinstance(term_names, dict): term_names = term_names.keys()

        for term_name in term_names:
            term = action_man._terms[term_name]
            dim = term.action_dim
            
            # === 核心逻辑：判断是否需要展开 ===
            # 如果这个 Term 有物理关节 ID，并且 关节数量 == 动作维度 这说明它是【一对一】控制 (即 Arm Joints)
            if hasattr(term, "_joint_ids") and len(term._joint_ids) == dim:
                for i, phys_idx in enumerate(term._joint_ids):
                    real_name = all_joint_names[phys_idx] 
                    mapping[real_name] = current_idx + i
            
            # === 其他情况 (夹爪) ===
            # 夹爪是【一对多】(1个动作控2个关节)，或者是没有关节ID的 我们直接用 Term 名字作为 Key
            else:
                mapping[term_name] = current_idx

            current_idx += dim
            
        return mapping