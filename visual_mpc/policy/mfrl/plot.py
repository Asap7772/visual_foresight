import numpy as np
import random
import torch
import ipdb
from PIL import Image
from rlkit.torch.core import eval_np
import rlkit.torch.pytorch_util as ptu
import io
from numpy import *
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
from pathlib import Path
import h5py
import ipdb
import os
import pickle
from PIL import Image

def enable_gpus(gpu_str):
    if gpu_str is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

def add_arrow(axs, state, action, c = 'b'):
	# action *= 0.02
	action = action/ np.linalg.norm(action) * 0.04
	# larg = np.argmax(action)
	# for i in range(action.shape[0]):
		# if i != larg:
			# action[i] = 0
	#xy
	axs[0].arrow(state[0],state[1],action[0], action[1], head_width=0.01, head_length=0.01, fc=c, ec=c)
	#xz
	axs[1].arrow(state[0],state[2],action[0], action[2], head_width=0.01, head_length=0.01, fc=c, ec=c)
	#yz
	axs[2].arrow(state[1],state[2],action[1], action[2], head_width=0.01, head_length=0.01, fc=c, ec=c)

def load_traj(path):
	all_images_cam0 = []
	all_images_cam1 = []
	all_states = []
	all_actions = []

	all_next_states = []
	all_next_images_cam0 = []
	all_next_images_cam1 = []
	all_next_actions = []

	curr = path
	obs = curr + "obs_dict.pkl"
	agent = curr + "agent_data.pkl"
	policy = curr + "policy_out.pkl"
	pimg0 = curr + "images0/im_"
	pimg1 = curr + "images1/im_"

	f_obs = open(obs, 'rb')
	state_dict = pickle.load(f_obs, encoding='latin1')
	states = np.concatenate((state_dict['state'] , state_dict['qpos'] , state_dict['qvel']), axis=1)

	f_pol = open(policy, 'rb')
	actions_lst = pickle.load(f_pol, encoding='latin1')
	tup = tuple(map(lambda x: x['actions'][None], actions_lst))
	actions = np.concatenate(tup, axis=0)
	img0_arr = []
	img1_arr = []

	for i in range(states.shape[0]):
		path_img0 = pimg0 + str(i) + '.jpg'
		path_img1 = pimg1 + str(i) + '.jpg'

		img0_arr.append(np.rollaxis(np.asarray(Image.open(path_img0).resize((64,48))),2)[None])
		img1_arr.append(np.rollaxis(np.asarray(Image.open(path_img1).resize((64,48))),2)[None])
		# import ipdb; ipdb.set_trace()

	images_cam0 = np.concatenate(img0_arr, axis=0)
	images_cam1 = np.concatenate(img1_arr, axis=0)

	all_actions.append(actions)
	all_states.append(states[:-1])
	all_images_cam0.append(images_cam0[:-1])
	all_images_cam1.append(images_cam1[:-1])

	all_next_states.append(states[1:])
	all_next_images_cam0.append(images_cam0[1:])
	all_next_images_cam1.append(images_cam1[1:])
	all_next_actions.append(np.concatenate((actions[1:],np.zeros(actions[0].shape)[None]), axis = 0))

	s = np.concatenate(all_states, axis = 0)
	a = np.concatenate(all_actions, axis = 0)
	i0 = np.concatenate(all_images_cam0, axis = 0)
	i1 = np.concatenate(all_images_cam1, axis = 0)
	ns = np.concatenate(all_next_states, axis= 0)
	ni0 = np.concatenate(all_next_images_cam0, axis= 0)
	ni1 = np.concatenate(all_next_images_cam1, axis= 0)
	na = np.concatenate(all_next_actions, axis= 0)
	print(a)
	return s, i0, i1, a, ns, ni0, ni1, na

if __name__ == "__main__":
	#traj_path = '/home/asap7772/0626_reaching/lag_10_dense/reach_test_1/traj_data'
	#param_path = '/home/asap7772/batch_rl_private/data/lagrange-20-widowx-full/455434/lagrange_20_widowx_full/455434_2020_06_14_21_15_33_0000--s-0'
	traj_path = '/home/stephentian/widow_ctrl/private_visual_foresight/experiments/mfrl/widow_reaching/widowx/lag_2_dense/reach_test_1/traj_data'
	#param_path = '/home/stephentian/widow_ctrl/private_visual_foresight/models/mfrl_reaching/lagrange-10-widowx-full/660764/lagrange_10_widowx_full/660764_2020_06_14_21_15_27_0000--s-0'
	#traj_path = '/home/stephentian/widow_ctrl/private_visual_foresight/experiments/mfrl/widow_reaching/widowx/lag_20_dense/reach_test_1/traj_data'
	param_path = '/home/asap7772/batch_rl_private/data/lagrange-2-widowx-full/44397/lagrange_2_widowx_full/44397_2020_06_14_21_15_07_0000--s-0'
	traj_path += '/'

	conv = True
	trunc = True
	pred = False
	goal = True
	transfer = False
	gx,gy,gz = 0.517103 , 0.587383 , 0.746937
	gx, gy, gz = 0.530896, 0.49834, 0.989866
	gx, gy, gz = 0.53096, 0.49834, 0.98
	gp = gx,gy,gz
	param_path += '/params.pkl'
	output_path = '/home/stephentian/'

	Path(output_path).mkdir(parents=True, exist_ok=True)
	output_path += '/'

	enable_gpus('2')
	ptu.set_gpu_mode(False)
	data = torch.load(param_path, map_location='cpu')

	policy = data['evaluation/policy'].stochastic_policy
	#policy.cpu()
	policy.eval()
	import ipdb; ipdb.set_trace()
	fig, axs = plt.subplots(3)
	
	axs[0].set_xlim([0,1.5])
	axs[0].set_ylim([0,1.5])
	axs[1].set_xlim([0,1.5])
	axs[1].set_ylim([0,1.5])
	axs[2].set_xlim([0,1.5])
	axs[2].set_ylim([0,1.5])

	axs[0].set_xlabel('x')
	axs[0].set_ylabel('y')
	axs[1].set_xlabel('x')
	axs[1].set_ylabel('z')
	axs[2].set_xlabel('y')
	axs[2].set_ylabel('z')
	
	if goal:
		axs[0].plot(gx,gy,'gp', markersize=14)
		axs[1].plot(gx,gz,'gp', markersize=14)
		axs[2].plot(gy,gz,'gp', markersize=14)

	observations, imgs, _, actions, next_obs, next_imgs, _, next_actions = load_traj(traj_path)
	
	# print(observations[:, :3])
	# print(actions)
	
	for i in range(imgs.shape[0]):
		state = observations[i]
		next_state = next_obs[i]
		img = imgs[i]
		action = actions[i]
		if transfer:
			state = np.concatenate((state[:3], np.random.uniform(low=0, high=1, size=(2,))))
		if pred:
			with torch.no_grad():
				img_x = np.expand_dims(img, axis=0)
				state = np.expand_dims(state, axis=0)
				if conv:
					state_x = None if trunc else state
					pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(policy, img_x, state_x, actions=None, reparameterize=True, deterministic=True, return_log_prob=False)
				else:
					pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(policy, state, reparameterize=True, deterministic=True, return_log_prob=False)
			state = np.squeeze(state)
			pred_action = np.squeeze(pred_action)
			print(state[:3], pred_action[:3])
			add_arrow(axs, state[:3], pred_action[:3], c = 'b')
			# ideal = (gp-state[:3])/np.linalg.norm(gp-state[:3])
			# add_arrow(axs, state[:3], ideal, c = 'y')
		else:
			add_arrow(axs, state[:3], action[:3], c = 'r')
			diff = next_state[:3] - state[:3]
			print(state[:3], action[:3])
			# add_arrow(axs, state[:3], diff, c = 'y')
			# add_arrow(axs, state[:3], -state[:3]/np.linalg.norm(state[:3])*0.02, c = 'y')
		buf = io.BytesIO()

	fig.set_size_inches(5.5, 9, forward=True)
	fig.tight_layout()
	plt.draw()

	plt.savefig(buf, format='png')
	buf.seek(0)
	img = Image.open(buf)
	img.show()
	