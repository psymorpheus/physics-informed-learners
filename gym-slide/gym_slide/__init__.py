from gym.envs.registration import register

register(
    id='slide-v0',
    entry_point='gym_slide.envs:SlideEnv',
)