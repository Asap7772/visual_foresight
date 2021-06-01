""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.interactive.keyboard_teleop import KeyboardTeleop
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.vanilla_env import VanillaEnv
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'camera_topics': [IMTopic('/front/image_raw')],
    'gripper_attached': 'wsg-50',
    'rand_drop_reset': False,
    'save_video': True,
    'cleanup_rate': -1,
    'start_at_neutral': True
}

agent = {
    'type': GeneralAgent,
    'env': (AutograspEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 240,
    'image_width' : 320,
    'duration': 0.01
}

policy = {
    'type': KeyboardTeleop
}

config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
