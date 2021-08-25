import os
import numpy as np

from gym import utils
from mujoco_collection_1.envs import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "mujoco_slide.xml")


class MujocoSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, obj_init, reward_type="sparse"):
        # First 3 tell coordinates of object and last 3 tell velocity
        assert obj_init.shape == (6,)
        # initial_qpos = {
        #     "robot0:slide0": 0.05,
        #     "robot0:slide1": 0.48,
        #     "robot0:slide2": 0.0,
        #     "object0:joint": [1.7, 1.1, 0.41, 1.0, 0.0, 0.0, 0.0],
        # }
        initial_qpos = {
            "object0:joint": [1.7, 1.1, 0.41, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=-0.02,
            target_in_the_air=False,
            target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1,
            target_range=0.3,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            obj_init=obj_init,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)
