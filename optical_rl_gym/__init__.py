from gym.envs.registration import register

register(
    id="RMSA-v0",
    entry_point="optical_rl_gym.envs:RMSAEnv",
)

register(
    id="DeepRMSA-v0",
    entry_point="optical_rl_gym.envs:DeepRMSAEnv",
)

