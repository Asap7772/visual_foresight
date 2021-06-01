import ipdb
import torch
import railrl.torch.pytorch_util as ptu
from railrl.torch.core import eval_np
import numpy as np
from visual_mpc.policy.policy import Policy
from torch import nn
import os
from visual_mpc.policy.data_augs import random_crop, random_convolution, random_color_jitter
from visual_mpc.policy.TransformLayer import ColorJitterLayer
import h5py
import pickle
import cv2


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class RandomModule(nn.Module):
    def __init__(self, wrap_module, p=1.0):
        super(RandomModule, self).__init__()
        self.wrap_module = wrap_module
        self.p = p

    def forward(self, inputs):
        if np.random.random() < self.p:
            return self.wrap_module(inputs)
        return inputs
    
class PolicyNet(nn.Module):
    
    def __init__(self, aug):
        super(PolicyNet, self).__init__()
        self.aug = aug
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(),
            nn.Linear(1024*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.reshape(-1, 64, 64, 3)
        x = x.permute(0, 3, 1, 2)
#         img = plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
#         plt.show()
        if self.aug:
            with HiddenPrints():
                x = augmentation_transform(x)
#                 img = plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
#                 plt.show()
        return self.cnn(x)


class BCPolicy(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.enable_gpus(str(gpu_id))
        #ptu.set_gpu_mode(True)
        p = PolicyNet(False)
        p.load_state_dict(torch.load(self._hp.path))
        p.eval()
        p = p.cuda()
        self.policy = p

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
        }
        parent_params = super(BCPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(BCPolicy, self).set_log_dir(d)

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
        gauss_mean = np.array([0, 0, -0.5, 0])
        #gauss_std = np.ones(4)
        #gauss_std = np.diag(np.array([0.3, 0.3, 0.3, 0.3]))
        gauss_std = np.diag(np.ones(4))
        for i in range(num_cem_iters):
            #acts = np.random.multivariate_normal(gauss_mean, np.diag(gauss_std**2), size=(num_cem_samps,))
            acts = np.random.multivariate_normal(gauss_mean, gauss_std, size=(num_cem_samps,))
            acts = acts.clip(min=(-1, -1, -1, -0.1), max=(1, 1, 1, 0.1))
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
        #img_x = np.transpose(stacked_images, (2, 0, 1)).flatten()
        img_x = stacked_images.flatten()
        img_x = np.expand_dims(img_x, axis=0) / 255.
        
        pred_action = self.policy(torch.FloatTensor(img_x).cuda()).detach().cpu().numpy()
        pred_action = np.squeeze(pred_action)
        print(pred_action)
        #pred_action *= 3.5 * np.array([0.02303233, 0.02453148, 0.03293544, 0.50120825])
        pred_action *= 5

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
    data = pickle.load(open('/home/stephen/offline_rl_models/auglag20/params_qfn_py2.pkl', 'rb'))
    policy = data['evaluation/policy'].stochastic_policy
    policy.eval()
    q_fn = data['trainer/trainer'].qf1
    q_fn.eval()

    img_x = load_png_image('/home/stephen/sawyer_setup_050321.png')
    img_y = load_png_image('/home/stephen/sawyer_setup_050321-2.png')
    pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value, _ = eval_np(
                    policy, img_x, None, deterministic=True,
                    return_log_prob=False)
    qval = get_maxqval(q_fn, policy, img_x)
    pred_action2, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value, _ = eval_np(
                    policy, img_y, None, deterministic=True,
                    return_log_prob=False)
    qval2 = get_maxqval(q_fn, policy, img_y)
    import ipdb; ipdb.set_trace()

