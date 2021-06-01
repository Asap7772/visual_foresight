import numpy as np
import os
#from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.mfrl.policy_wrapper import RLPolicy
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])


env_params = {
    # 'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/kinect2/qhd/image_color'),
                      IMTopic('/front/image_raw', flip=True)],
    #                 IMTopic('/right/image_raw')],
    'robot_name': 'widowx',
    'robot_type': 'widowx',
    'gripper_attached': 'default',
    'OFFSET_TOL': 3,
    'robot_upside_down': True,
    'zthresh': 0.2,
    'gripper_joint_thresh': 0.85,
    'gripper_joint_grasp_min': -0.9,
    'cleanup_rate': 12,
    'normalize_actions': True,
}

agent = {'type': GeneralAgent,
         'env': (AutograspEnv, env_params),
         'T': 30,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'data_save_dir': BASE_DIR,
         'log_dir': '/root/ros_ws/src/brl_private_visual_foresight/'
         #'image_height': 48,
         #'image_width': 64,
}

policy = {
    'type': RLPolicy,
    'log': True,
    #'user': True,
    'path': '/root/ros_ws/src/brl_private_visual_foresight/models/data/lagrange-10-widowx-full/660764/lagrange_10_widowx_full/660764_2020_06_14_21_15_27_0000--s-0/params.pkl',
    'policy_type': 1,
    #'goal_pos': (0.935024, 0.204873, 0.0694792),
}

config = {
    "experiment_name": "lag_10_dense",
    'traj_per_file':128,
    'save_data': True,
    'save_raw_images' : True,
    'start_index':0,
    'end_index': 30000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000,
    'nshuffle' : 200
}
