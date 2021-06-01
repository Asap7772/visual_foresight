import numpy as np
import os
#from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.mfrl.bc_policy import BCPolicy
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
#

env_params = {
    'robot_name': 'galar',
    'camera_topics': [IMTopic('/front/image_raw', flip=False)],                  #, IMTopic('/bot/image_raw'), IMTopic('/bot2/image_raw')],
    'cleanup_rate': -1,
    'save_video': True,
    'rand_drop_reset': False,
    'normalize_actions': False,
}

# env_params = {
#     # 'email_login_creds': '.email_cred',
#     'camera_topics': [IMTopic('/kinect2/qhd/image_color'),
#                       IMTopic('/front/image_raw', flip=True)],
#     #                 IMTopic('/right/image_raw')],
#     'robot_name': 'widowx',
#     'robot_type': 'widowx',
#     'gripper_attached': 'default',
#     'OFFSET_TOL': 3,
#     'robot_upside_down': True,
#     'zthresh': 0.2,
#     'gripper_joint_thresh': 0.85,
#     'gripper_joint_grasp_min': -0.9,
#     'cleanup_rate': 12,
#     'normalize_actions': True,
# }

agent = {'type': GeneralAgent,
         'env': (AutograspEnv, env_params),
         'T': 15,  #number of commands per episodes (issued at control_rate / substeps HZ)
         'data_save_dir': BASE_DIR,
         'log_dir': '/home/stephen/ros_ws/src/private_visual_foresight/outputs',
         'image_height': 64,
#'image_width': 64,
}

policy = {
    'type': BCPolicy,
    'log': True,
    'user': True,
    "path": '/home/stephen/offline_rl_models/bc_policy_aug_all.pkl',
#    'path': '/nfs/kun1/users/asap7772/real_data_tooluse/data/real-lagr1-bottleneck/real_lagr1_bottleneck_2021_05_24_19_13_46_id834099--s0/params_v3.pkl',
#'optimize_q_function': True,
    'policy_type': 1,
    #'goal_pos': (0.935024, 0.204873, 0.0694792),
}

config = {
    "experiment_name": "offline_tool_aug_qmax_5_19",
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
