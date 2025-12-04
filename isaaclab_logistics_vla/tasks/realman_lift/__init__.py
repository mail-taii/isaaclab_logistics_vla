import gymnasium as gym


gym.register(
    id="Isaac-Realman-lift",
    
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    
    disable_env_checker=True,
    
    kwargs={
        "env_cfg_entry_point": f"{__name__}.realman_lift_env_cfg:LiftEnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RealmanPPORunnerCfg",
    },
)