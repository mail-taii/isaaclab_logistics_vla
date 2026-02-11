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

            joint_ids = getattr(term, "_joint_ids", None)
            if joint_ids is not None:
                # _joint_ids 可能是 list/array 或 slice（如 UR5e 的 JointPositionAction）
                if isinstance(joint_ids, slice):
                    joint_ids = list(range(*joint_ids.indices(len(all_joint_names))))
                n_joints = len(joint_ids) if hasattr(joint_ids, "__len__") else 0
            else:
                n_joints = 0

            # === 核心逻辑：判断是否需要展开 ===
            # 如果这个 Term 有物理关节 ID，并且 关节数量 == 动作维度 这说明它是【一对一】控制 (即 Arm Joints)
            if joint_ids is not None and n_joints == dim:
                for i, phys_idx in enumerate(joint_ids):
                    real_name = all_joint_names[phys_idx]
                    mapping[real_name] = current_idx + i
            # === 其他情况 (夹爪 / 多关节一个 term 等) ===
            else:
                mapping[term_name] = current_idx

            current_idx += dim
            
        return mapping