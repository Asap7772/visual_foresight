""" This file defines the linear Gaussian policy class. """
from visual_mpc.utils.im_utils import npy_to_gif
import copy
from visual_mpc.policy.cem_controllers.cem_base_controller import CEMBaseController
import numpy as np
import matplotlib.pyplot as plt
from visual_mpc.utils.im_utils import resize_store
import ray
import traceback
import pdb



class SimWorker(object):
    def __init__(self):
        print('created worker')
        pass

    def create_sim(self, agentparams, reset_state, finalweight, len_pred, verbose, goal_obj_pose, goal_arm_pose):
        print('create sim') #todo
        self.verbose = verbose
        self.agentparams = agentparams
        self.len_pred = len_pred
        self.finalweight = finalweight
        self.current_reset_state = reset_state

        env_type, env_params = self.agentparams['env']
        # env_params['verbose_dir'] = '/home/frederik/Desktop/'
        self.env = env_type(env_params, self.current_reset_state)
        self.env.set_goal(goal_obj_pose, goal_arm_pose)

        # hyperparams passed into sample_action function
        # class HP(object):
        #     def __init__(self, naction_steps, discrete_ind, action_bound, adim, initial_std):
        #         self.naction_steps = naction_steps
        #         self.discrete_ind = discrete_ind
        #         self.action_bound = action_bound
        #         self.adim = adim
        #         self.initial_std = initial_std
        # self.hp = HP(naction_steps, discrete_ind, action_bound, adim, initial_std)
        print('finished creating sim')

    def eval_action(self):
        return self.env.get_distance_score()

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        agent_img_height = self.agentparams['image_height']
        agent_img_width = self.agentparams['image_width']

        if initial_obs:
            T = self.len_pred + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    n_cams = env_obs['images'].shape[0]
                    self._agent_cache['images'] = np.zeros((T, n_cams, agent_img_height, agent_img_width, 3),
                                                           dtype = np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = float(self.agentparams.get('point_space_width', agent_img_width))
        obs = {}
        for k in env_obs:
            if k == 'images':
                resize_store(t, self._agent_cache['images'], env_obs['images'])
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  #only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / agent_img_width)).astype(np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        return obs

    def sim_rollout(self, curr_qpos, curr_qvel, actions):
        agent_data = {}
        t = 0
        done = False
        reset_state = {'qpos_all':curr_qpos, 'qvel_all':curr_qvel}
        initial_env_obs, _ = self.env.reset(reset_state)

        print('############')
        print('cem ctrl sim:')
        print('curr qpos', curr_qpos)
        print('curr qvel', curr_qvel)
        print('resetted to qpos', initial_env_obs['qpos_full'])
        print('resetted qvel', initial_env_obs['qvel_full'])
        print('############')

        obs = self._post_process_obs(initial_env_obs, agent_data, initial_obs=True)
        costs = []
        while not done:
            obs = self._post_process_obs(self.env.step(actions[t], render=self.verbose), agent_data)
            if (self.len_pred - 1) == t:
                done = True
            t += 1
            costs.append(self.eval_action())

        if not self.verbose:
            agent_img_height = self.agentparams['image_height']
            agent_img_width = self.agentparams['image_width']
            obs['images'] = np.zeros((self.len_pred, 1, agent_img_height, agent_img_width))

        return costs, obs['images']

    def perform_rollouts(self, curr_qpos, curr_qvel, actions, M):
        all_scores = np.empty(M, dtype=np.float64)
        image_list = []


        for smp in range(M):
            score, images = self.sim_rollout(curr_qpos, curr_qvel, actions[smp])

            image_list.append(images.squeeze())
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self.finalweight
            all_scores[smp] = np.sum(per_time_multiplier*score)

        images = np.stack(image_list, 0)[:,1:].astype(np.float32)/255.

        return images, np.stack(all_scores, 0)


@ray.remote(num_gpus=1)
class ParallelSimWorker(SimWorker):
    def __init__(self):
        SimWorker.__init__(self)


class CEM_Controller_Sim(CEMBaseController):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)
        if self._hp.num_workers == 1:
            self.parallel = False
        else:
            self.parallel = True
            ray.init()

        self.len_pred = policyparams['nactions']

    def _default_hparams(self):
        default_dict = {
            'len_pred':15,
            'num_workers':10,
            'finalweight':10,
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def create_sim(self):
        self.workers = []
        if self.parallel:
            self.n_worker = self._hp.num_workers
        else:
            self.n_worker = 1

        if self.parallel:
            simworker_class = ParallelSimWorker
        else:
            simworker_class = SimWorker

        for i in range(self.n_worker):
            if self.parallel:
                self.workers.append(simworker_class.remote())
            else:
                self.workers.append(simworker_class())

        id_list = []
        for i, worker in enumerate(self.workers):
            if self.parallel:
                id_list.append(worker.create_sim.remote(self.agentparams, self.curr_sim_state, self._hp.finalweight, self.len_pred, self._hp.verbose, self.goal_obj_pose, self.goal_arm_pose))
            else:
                return worker.create_sim(self.agentparams, self.curr_sim_state, self._hp.finalweight, self.len_pred, self._hp.verbose, self.goal_obj_pose, self.goal_arm_pose)
        if self.parallel:
            # blocking call
            for id in id_list:
                ray.get(id)

    def evaluate_rollouts(self, actions, cem_itr):
        images, all_scores = self.sim_rollout_parallel(actions)

        if 'gif_freq' in self.agentparams:
            if self.i_tr % self.agentparams['gif_freq'] == 0:
                gif_ok = True
            else: gif_ok = False
        else: gif_ok = True
        if self._hp.verbose and gif_ok:
            self.save_gif(images, all_scores, cem_itr)
        return all_scores

    def save_gif(self, images, all_scores, cem_itr):
        bestindices = all_scores.argsort()[:self.K]
        images = (images[bestindices]*255.).astype(np.uint8)  # select cam0
        vid = []
        for t in range(self._hp.nactions):
            row = np.concatenate(np.split(images[:,t], images.shape[0], axis=0), axis=2).squeeze()
            vid.append(row)

        vid.append(np.zeros_like(vid[0]))

        name = 't{}_iter{}'.format(self.t, cem_itr)
        npy_to_gif(vid, self.traj_log_dir + '/' + name)


    def sim_rollout_parallel(self, actions):
        per_worker = int(self._hp.num_samples / np.float32(self.n_worker))
        id_list = []
        for i, worker in enumerate(self.workers):
            if self.parallel:
                actions_perworker = actions[i*per_worker:(i+1)*per_worker]
                id_list.append(worker.perform_rollouts.remote(self.qpos_full, self.qvel_full, actions_perworker, per_worker))
            else:
                images, scores_mjc = worker.perform_rollouts(self.qpos_full, self.qvel_full, actions, self._hp.num_samples)

        # blocking call
        if self.parallel:
            image_list, scores_list = [], []
            for id in id_list:
                images, scores_mjc = ray.get(id)
                image_list.append(images)
                scores_list.append(scores_mjc)
            scores_mjc = np.concatenate(scores_list, axis=0)
            images = np.concatenate(image_list, axis=0)

        scores = self.get_scores(images, scores_mjc)
        return images, scores

    def get_scores(self, images, scores_mjc):
        return scores_mjc

    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self.agentparams['substeps']
        return substep/float(self.agentparams['substeps'])*(next - prev) + prev

    def plot_ctrls(self):
        plt.figure()
        # a = plt.gca()
        self.hf_qpos_l = np.stack(self.hf_qpos_l, axis=0)
        self.hf_target_qpos_l = np.stack(self.hf_target_qpos_l, axis=0)
        tmax = self.hf_target_qpos_l.shape[0]
        for i in range(self.adim):
            plt.plot(list(range(tmax)) , self.hf_qpos_l[:,i], label='q_{}'.format(i))
            plt.plot(list(range(tmax)) , self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
            plt.legend()
            plt.show()

    def act(self, t, i_tr, qpos_full, qvel_full, state, object_qpos, reset_state, goal_obj_pose, goal_arm_pose):
        self.curr_sim_state = reset_state
        if len(qpos_full.shape) == 2:  # if qpos contains time
            self.qpos_full = qpos_full[t]
            self.qvel_full = qvel_full[t]
        else:
            self.qpos_full = qpos_full
            self.qvel_full = qvel_full

        self.goal_obj_pose = goal_obj_pose
        self.goal_arm_pose = goal_arm_pose

        self.i_tr = i_tr

        self.t = t
        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(t, i_tr)


