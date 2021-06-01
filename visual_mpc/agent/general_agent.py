""" This file defines an agent for the MuJoCo simulator environment. """
import pdb
import copy
import numpy as np
import os
from visual_mpc.policy import get_policy_args
from visual_mpc.utils.im_utils import resize_store
from .utils.file_saver import start_file_worker
from visual_mpc.utils.im_utils import npy_to_gif
import cv2

from tensorflow.contrib.training import HParams


class Bad_Traj_Exception(Exception):
    def __init__(self):
        pass


class Image_Exception(Exception):
    def __init__(self):
        pass


class Environment_Exception(Exception):
    def __init__(self):
        pass


class GeneralAgent(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """

    def __init__(self, hyperparams):
        self._hp = self._default_hparams()
        self.override_defaults(hyperparams)

        self.T = self._hp.T
        self._goal = None
        self._goal_seq = None
        self._goal_image = None
        self._demo_images = None
        self._reset_state = None
        self._is_robot = 'robot_name' in hyperparams['env'][1]
        if self._hp.use_save_thread:
            self._save_worker = start_file_worker()
        self._setup_world(0)

    def override_defaults(self, config):
        """
        :param config:  override default valus with config dict
        :return:
        """
        for name, value in config.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute is {} is identical to default value!!".format(name))
            elif isinstance(value, tuple):   # don't do a type check for lists
                setattr(self._hp, name, value)
            elif name in self._hp and self._hp.get(name) is None:   # don't do a type check for None default values
                setattr(self._hp, name, value)
            else: self._hp.set_hparam(name, value)

    def _default_hparams(self):
        default_dict = {
            'T':None,
            'adim':None,
            'sdim':None,
            'ncam':1,
            'rejection_sample':False,   # repeatedly attemp to collect a trajectory if error occurs
            'type':None,
            'env':None,
            'image_height' : 48,
            'image_width' : 64,
            'nchannels':3,
            'data_save_dir':'',     # path where collected training data will be stored
            'log_dir':'',           # path where logs and viusals will be stored
            'make_final_gif':True,   # whether to make final gif
            'make_final_gif_freq':1,   # final gif, frequency
            'make_final_gif_pointoverlay':False,
            'gen_xml': (True, 1),  # whether to generate xml, and how often
            'start_goal_confs': None,
            'use_save_thread':False,
            'num_load_steps':2,
            'robot_name': None,
            'imax': 0
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params = self._hp.env
        self.env = env_type(env_params)

        self._hp.adim = self.adim = self.env.adim
        self._hp.sdim = self.sdim = self.env.sdim
        self._hp.ncam = self.ncam = self.env.ncam
        self.num_objects = self.env.num_objects

    def reset_env(self):
        initial_env_obs, self._reset_state = self.env.reset()
        return initial_env_obs

    def sample(self, policy, i_traj):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        self.i_traj = i_traj
        if self._hp.gen_xml[0]:
            if i_traj % self._hp.gen_xml[1] == 0 and i_traj > 0:
                self._setup_world(i_traj)

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0
        imax = 100

        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial, i_traj)
                traj_ok = agent_data['traj_ok']

            except (Image_Exception, Environment_Exception):
                traj_ok = False

        if not traj_ok:
            raise Bad_Traj_Exception

        print('needed {} trials'.format(i_trial))

        if self._hp.make_final_gif or self._hp.make_final_gif_pointoverlay:
            if i_traj % self._hp.make_final_gif_freq == 0:
                # self._save_worker.put(('path', self.record_path))
                # self.env.save_recording(self._save_worker, i_traj)
                self.save_gif(i_traj)

        return agent_data, obs_dict, policy_outs

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        """
        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        Data accumlated over time is cached into an observation dict and returned. Data specific to each
        time-step is returned in agent_data

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """
        agent_img_height = self._hp.image_height
        agent_img_width = self._hp.image_width

        if initial_obs:
            T = self._hp.T + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    self._agent_cache['images'] = np.zeros((T, self._hp.ncam, agent_img_height, agent_img_width, self._hp.nchannels), dtype=np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = agent_img_width

        obs = {}
        for k in env_obs:
            if k == 'images':
                self.gif_images_traj.append(env_obs['images'][0])  # only take first camera
                resize_store(t, self._agent_cache['images'], env_obs['images'])

            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  # only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / env_obs['images'].shape[2])).astype(
                    np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        if self._goal_image is not None:
            agent_data['goal_image'] = self._goal_image
        if hasattr(self, '_goal_obj_pose') and self._goal_obj_pose is not None:
            agent_data['goal_obj_pose'] = self._goal_obj_pose.copy()
        if hasattr(self, '_goal_arm_pose') and self._goal_arm_pose is not None:
            agent_data['goal_arm_pose'] = self._goal_arm_pose.copy()
        if hasattr(self, '_goal_states') and self._goal_states is not None:
            agent_data['goal_states'] = self._goal_states.copy()
        if self._demo_images is not None:
            agent_data['demo_images'] = self._demo_images
        if self._reset_state is not None:
            agent_data['reset_state'] = self._reset_state
            obs['reset_state'] = self._reset_state

        return obs

    def _required_rollout_metadata(self, agent_data, t, traj_ok):
        """
        Adds meta_data into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        agent_data['term_t'] = t - 1


    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()

        agent_data, policy_outputs = {}, []

        # Take the sample.
        t = 0
        done = self._hp.T <= 0
        initial_env_obs = self.reset_env()
        obs = self._post_process_obs(initial_env_obs, agent_data, True)
        policy.reset()

        self.traj_log_dir = self._hp.log_dir + '/verbose/traj{}'.format(i_traj)
        if not os.path.exists(self.traj_log_dir):
            os.makedirs(self.traj_log_dir)
        policy.set_log_dir(self.traj_log_dir)

        while not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """
            pi_t = policy.act(**get_policy_args(policy, obs, t, i_traj, agent_data))
            policy_outputs.append(pi_t)

            if 'done' in pi_t:
                done = pi_t['done']
            try:
                obs = self._post_process_obs(self.env.step(pi_t['actions']), agent_data)
                # obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions']), stage=stage), agent_data, stage=pi_t['policy_index'])
            except Environment_Exception as e:
                print(e)
                return {'traj_ok': False}, None, None


            if (self._hp.T - 1) == t or 'env_done' in obs and obs['env_done'][-1]:   # environements can include the tag 'env_done' in the observations to signal that time is over
                done = True
            t += 1

        traj_ok = self.env.valid_rollout()
        if self._hp.rejection_sample:
            assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
            traj_ok = self.env.goal_reached()
            print('goal_reached', traj_ok)

        agent_data['traj_ok'] = traj_ok

        self._required_rollout_metadata(agent_data, t, traj_ok)

        return agent_data, obs, policy_outputs


    def save_gif(self, i_traj, overlay=False):
        if self.traj_points is not None and overlay:
            colors = [tuple([np.random.randint(0, 256) for _ in range(3)]) for __ in range(self.num_objects)]
            for pnts, img in zip(self.traj_points, self.gif_images_traj):
                for i in range(self.num_objects):
                    center = tuple([int(np.round(pnts[i, j])) for j in (1, 0)])
                    cv2.circle(img, center, 4, colors[i], -1)

        npy_to_gif(self.gif_images_traj, self.traj_log_dir + '/video'.format(i_traj)) # todo make extra folders for each run?

    def _init(self):
        """
        Set the world to a given model
        """
        self.gif_images_traj, self.traj_points = [], None

    def cleanup(self):
        if self._hp.use_save_thread:
            print('Cleaning up file saver....')
            self._save_worker.put(None)
            self._save_worker.join()

    @property
    def record_path(self):
        return self._hp.log_dir+ '/record/'
