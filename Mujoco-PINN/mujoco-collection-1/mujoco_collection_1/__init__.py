from gym.envs.registration import register

register(
    id='mujoco-hit-v0',
    entry_point='mujoco_collection_1.envs:MujocoHitEnv',
)
register(
    id='mujoco-slide-v0',
    entry_point='mujoco_collection_1.envs:MujocoSlideEnv',
)
