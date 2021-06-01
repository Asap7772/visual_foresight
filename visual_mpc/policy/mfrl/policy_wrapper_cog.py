import ipdb
import torch
import railrl.torch.pytorch_util as ptu
from railrl.torch.core import eval_np
import numpy as np
from visual_mpc.policy.policy import Policy
from torch import nn
import os
from visual_mpc.policy.data_augs import random_crop, random_convolution, random_color_jitter
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from visual_mpc.policy.TransformLayer import ColorJitterLayer
import h5py
import pickle
import cv2

class RLPolicyCOG(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.enable_gpus(str(gpu_id))
        ptu.set_gpu_mode(True)
        ptu.set_gpu_mode(True)
        parameters = torch.load(self._hp.path)
        action_dim = 4
        
        cnn_params=dict(
                    kernel_sizes=[3, 3, 3],
                    n_channels=[16, 16, 16],
                    strides=[1, 1, 1],
                    hidden_sizes=[1024, 512, 256],
                    paddings=[1, 1, 1],
                    pool_type='max2d',
                    pool_sizes=[2, 2, 1],
                    pool_strides=[2, 2, 1],
                    pool_paddings=[0, 0, 0],
                    image_augmentation=True,
                    image_augmentation_padding=4,
                    input_width=64,
                    input_height=64,
                    input_channels=3,
        )
        cnn_params.update(
                output_size=256,
                added_fc_input_size=0,
                hidden_sizes=[1024, 512],
        )
        policy_obs_processor = CNN(**cnn_params)
        self.policy = TanhGaussianPolicy(
            obs_dim=cnn_params['output_size'],
            action_dim=action_dim,
            hidden_sizes=[256, 256, 256],
            obs_processor=policy_obs_processor,
        )

        self.policy.load_state_dict(parameters['policy_state_dict'])
        if self._hp.bottleneck:
            self.qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64)
        else:
            cnn_params=dict(
                    kernel_sizes=[3, 3, 3],
                    n_channels=[16, 16, 16],
                    strides=[1, 1, 1],
                    hidden_sizes=[1024, 512, 256],
                    paddings=[1, 1, 1],
                    pool_type='max2d',
                    pool_sizes=[2, 2, 1],
                    pool_strides=[2, 2, 1],
                    pool_paddings=[0, 0, 0],
                    image_augmentation=True,
                    image_augmentation_padding=4,
            )
            cnn_params.update(
                    input_width=64,
                    input_height=64,
                    input_channels=3,
                    output_size=1,
                    added_fc_input_size=4,
            )
            self.qf1 = ConcatCNN(**cnn_params)
        self.qf1.load_state_dict(parameters['qf1_state_dict'])

    def enable_gpus(self, gpu_str):
        if gpu_str is not "":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    def _default_hparams(self):
        default_dict = {
            'user': False,
            'path': '/home/asap7772/batch_rl_private/data/lagrange-10-robonet-widowx/302109/lagrange_10_robonet_widowx/302109_2020_05_27_01_42_24_0000--s-0',
            'policy_type': 2,
            'goal_pos': (0.935024, 0.204873, 0.0694792),
            'data_aug': False,
            'data_aug_version': 0,
            'goals_path': '/raid/asap7772/3cam_widowx_data.hdf5',
            'log': False,
            'num_views': 1,
            'goal_cond': False,
            'goal_cond_version': 'gc_img',
            'optimize_q_function': False,
            'bottleneck': False,
        }
        parent_params = super(RLPolicyCOG, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(RLPolicyCOG, self).set_log_dir(d)

    def multiview_goal_imgs(self):
        rf = h5py.File(self._hp.goals_path, 'r')
        imgs = rf['images'][()]
        imgs1 = rf['images1'][()]
        imgs2 = rf['images2'][()]
        observations = rf['states'][()]
        actions = rf['actions'][()]
        next_imgs = rf['next_images'][()]
        next_observations = rf['next_states'][()]
        next_actions = rf['next_actions'][()]
        index = np.random.randint(0, high=imgs.shape[0]-1, size=1, dtype=int)[0]
        gx,gy,gz = observations[index][:3]
        self.goal_state = observations[index]
        self.goal_pos = (gx,gy,gz)
        print('----------------------')
        print('Goal pos:', gx, gy, gz)
        print('----------------------')
        self.gimages = imgs[index], imgs1[index], imgs2[index]

    def rand_crop(self, imgs):
        pad = nn.ReplicationPad2d(self.image_augmentation_padding)
        #crop = torchvision.transforms.RandomCrop(self.size[1:])
        from visual_mpc.policy.data_augs import random_crop
        return random_crop(pad(imgs), self.size[1:])

    def get_qval(self, q, obs, acts):
        return eval_np(q, obs, acts)
        
    def get_pred_action_cem(self, q, obs, num_cem_iters=7, num_cem_samps=600, cem_frac=0.1):
        #gauss_mean = np.zeros(4)
        gauss_mean = np.array([0, 0, 0, 0])
        #gauss_std = np.ones(4)
        #gauss_std = np.diag(np.array([0.3, 0.3, 0.3, 0.3]))
        gauss_std = np.diag(np.ones(4))
        for i in range(num_cem_iters):
            #acts = np.random.multivariate_normal(gauss_mean, np.diag(gauss_std**2), size=(num_cem_samps,))
            acts = np.random.multivariate_normal(gauss_mean, gauss_std, size=(num_cem_samps,))
            acts = acts.clip(min=(-1, -1, -1, -1), max=(1, 1, 1, 1))
            qvals = self.get_qval(q, np.repeat(obs.flatten()[None], acts.shape[0], 0), acts).squeeze()
            print('----- itr {} ----'.format(i))
            print('mean qval = {}'.format(qvals.mean()))
            print('std qval = {}'.format(qvals.std()))
            print('-----------------'.format(i))
            best_action_inds = (-qvals).argsort()[:int(num_cem_samps * cem_frac)]
            best_acts = acts[best_action_inds]
            gauss_mean = best_acts.mean(axis=0)
            #gauss_std = best_acts.std(axis=0)
            gauss_std = np.cov(best_acts, rowvar=False)
            print(gauss_std)
        print('cem choosing action', gauss_mean)
        print('q value of ', eval_np(q, obs, gauss_mean[None]))
        return gauss_mean

    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, verbose_worker=None):
        import ipdb; ipdb.set_trace()
        if t == 0 and self._hp.goal_cond:
            self.multiview_goal_imgs()

        if self._hp.log: self.file_path = os.path.join(self.traj_log_dir, 'log.txt')

        if self._hp.user:
            import ipdb
            ipdb.set_trace()

        if self._hp.goal_cond:
            print('error: ', np.array(self.goal_pos) - state[-1].squeeze()[:3])

        state = np.expand_dims(state[-1], axis=0)
        print(len(images))
        recent_img = images[-1]
        stacked_images = [recent_img[i] for i in range(self._hp.num_views)]
        stacked_images = np.concatenate(stacked_images, axis=2)
        img_x = np.transpose(stacked_images, (2, 0, 1)).flatten()
        img_x = stacked_images.flatten()
        img_x = np.expand_dims(img_x, axis=0) / 255.

        if self._hp.optimize_q_function:
            pred_action = self.get_pred_action_cem(self.q_function, img_x)
        else:
            with torch.no_grad():
                # img_x = img_x.transpose((0, 3, 1, 2)) / 255.
                # img_x = np.zeros((1, 3, 48, 64))

                if self._hp.goal_cond:
                    if self._hp.goal_cond_version == 'multiview':
                        self._hp.policy_type = 1
                        img_x = np.squeeze(img_x)
                        #self.gimages is a tuple of 3 images of shape (3, 48, 64)
                        img_x = np.concatenate((img_x,) + self.gimages)
                        img_x = img_x[None]
                    elif self._hp.goal_cond_version == 'gc_img':
                        self._hp.policy_type = 1
                        img_x = np.squeeze(img_x)
                        img_x = np.concatenate((img_x,) + self.gimages[:1])
                        img_x = img_x[None]
                    elif self._hp.goal_cond_version == 'gc_state':
                        self._hp.policy_type = 2
                        state = self.goal_state[None]

                if self._hp.data_aug:
                    # need to have kornia v0.2.0
                    # assuming size is (1, 3, 48, 64) but should work even if (3, 48, 64)
                    imgs = torch.from_numpy(np.squeeze(img_x)[None]) # just ensuring that batch size dim exists
                    imgs = imgs.to(torch.device("cuda"))
                    self.image_augmentation_padding = 4
                    self.size = np.array(tuple(imgs.shape[1:]))
                    imgs = imgs.type(torch.FloatTensor)
                    if self._hp.data_aug_version == 0:
                        imgs = self.rand_crop(imgs)
                    elif self._hp.data_aug_version == 1:
                        imgs = self.rand_crop(imgs)
                        imgs = random_convolution(imgs)
                    elif self._hp.data_aug_version == 2:
                        color_jitter = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5)
                        imgs = self.rand_crop(imgs)
                        imgs = color_jitter(imgs)
                    else:
                        pass
                    img_x = torch.squeeze(imgs)[None].cpu().numpy()
        import ipdb; ipdb.set_trace()
        pred_action = self.policy.get_action(img_x.squeeze())
        pred_action = np.squeeze(pred_action[0])
        print(pred_action)
        #pred_action *= 3.5 * np.array([0.02303233, 0.02453148, 0.03293544, 0.50120825])
        pred_action = np.array([pred_action[0], pred_action[1], pred_action[2], 0, pred_action[3]])

        if self._hp.log:
            file = open(self.file_path, 'a+')
            file.write(str(pred_action) + '\n')
            #file.write(str(log_prob) + '\n')
            file.close()
        print('----------------------')
        print(pred_action)
        print('----------------------')
        policy_out = {'actions': pred_action}
        if self._hp.goal_cond:
            from copy import deepcopy
            policy_out['goal_pos'] = deepcopy(self.goal_pos)
            policy_out['goal_ims'] = deepcopy(self.gimages)

        return policy_out



def get_maxqval(q, policy, obs):
    pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value , _= eval_np(policy, obs, None, deterministic=True, return_log_prob=False)
    #np.random.shuffle(pred_action)
    return eval_np(q, obs, pred_action)

def load_png_image(path):
    img_x = cv2.imread(path) 
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.
    img_x = cv2.resize(img_x, (64, 64), interpolation=cv2.INTER_AREA).flatten()[None]
    return img_x

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    parameters = torch.load('/home/stephen/eval_pickles_528/pt_drawer/drawer_minq1.pt')
    action_dim = 4
    
    cnn_params=dict(
                kernel_sizes=[3, 3, 3],
                n_channels=[16, 16, 16],
                strides=[1, 1, 1],
                hidden_sizes=[1024, 512, 256],
                paddings=[1, 1, 1],
                pool_type='max2d',
                pool_sizes=[2, 2, 1],
                pool_strides=[2, 2, 1],
                pool_paddings=[0, 0, 0],
                image_augmentation=True,
                image_augmentation_padding=4,
                input_width=64,
                input_height=64,
                input_channels=3,
    )
    cnn_params.update(
            output_size=256,
            added_fc_input_size=0,
            hidden_sizes=[1024, 512],
    )
    policy_obs_processor = CNN(**cnn_params)
    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    policy.load_state_dict(parameters['policy_state_dict'])
    if True:
        qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64)
    else:
        cnn_params=dict(
                kernel_sizes=[3, 3, 3],
                n_channels=[16, 16, 16],
                strides=[1, 1, 1],
                hidden_sizes=[1024, 512, 256],
                paddings=[1, 1, 1],
                pool_type='max2d',
                pool_sizes=[2, 2, 1],
                pool_strides=[2, 2, 1],
                pool_paddings=[0, 0, 0],
                image_augmentation=True,
                image_augmentation_padding=4,
        )
        cnn_params.update(
                input_width=64,
                input_height=64,
                input_channels=3,
                output_size=1,
                added_fc_input_size=4,
        )
        qf1 = ConcatCNN(**cnn_params)
    qf1.load_state_dict(parameters['qf1_state_dict'])

    import ipdb; ipdb.set_trace()

