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
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(),
            nn.Linear(65536, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.action_encoder = nn.Linear(4, 64)
        self.LSTM_cell = nn.LSTMCell(64, 256)
        self.out_embedding = nn.Linear(256, 10*8)
        self.conditioner = nn.Linear(261, 512)

    def forward(self, x, actions, init_state):
        x = x.reshape(-1, 64, 64, 3)
        x = x.permute(0, 3, 1, 2)
#         img = plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
#         plt.show()
        if self.aug:
            with HiddenPrints():
                x = augmentation_transform(x)
#                 img = plt.imshow(np.transpose(x[0].cpu().numpy(), (1, 2, 0)))
#                 plt.show()
        x_embed = self.cnn(x)
        x_embed = torch.cat((x_embed, init_state), dim=-1)
        condition = self.conditioner(x_embed)
        hx, cx = condition[:, :256], condition[:, 256:]
        outs = []
        for i in range(actions.shape[1]):
            if i == 0:
                hx, cx = self.LSTM_cell(self.action_encoder(torch.zeros_like(actions[:, 0]).cuda()), (hx, cx))
            else:
                hx, cx = self.LSTM_cell(self.action_encoder(actions[:, i-1]), (hx, cx))
            out_act_embed = self.out_embedding(hx)
            out_act_embed = out_act_embed.reshape(-1, 10, 8)
            dist = torch.distributions.independent.Independent(torch.distributions.normal.Normal(out_act_embed[..., :4], torch.exp(out_act_embed[..., 4:])), 1)
            mix = torch.distributions.mixture_same_family.MixtureSameFamily(torch.distributions.categorical.Categorical(torch.ones(10,).cuda()), dist)
            outs.append(mix)
        return outs
    
    def get_test_seq(self, x, init_state, nactions=15):
        x = x.reshape(-1, 64, 64, 3)
        x = x.permute(0, 3, 1, 2)
        x_embed = self.cnn(x)
        x_embed = torch.cat((x_embed, init_state), dim=-1)
        condition = self.conditioner(x_embed)
        hx, cx = condition[:, :256], condition[:, 256:]
        outs = []
        prev_action = None
        for i in range(nactions):
            if i == 0:
                hx, cx = self.LSTM_cell(self.action_encoder(torch.zeros(1, 4).cuda()), (hx, cx))
            else:
                hx, cx = self.LSTM_cell(self.action_encoder(prev_action), (hx, cx))
            out_act_embed = self.out_embedding(hx)
            out_act_embed = out_act_embed.reshape(-1, 10, 8)
            rand_int = np.random.randint(10)
            dist = torch.distributions.normal.Normal(out_act_embed[:, rand_int, :4], torch.exp(out_act_embed[:, rand_int, 4:])+1e-7)
            act = dist.sample()
            outs.append(act)
            prev_action = act
        print('action sequence', outs)
        return outs
    
    

class RecurrBCPolicy(Policy):
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
        parent_params = super(RecurrBCPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(RecurrBCPolicy, self).set_log_dir(d)

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
        if t == 0:
            state[:, -1] *= -1 
            self.actions = self.policy.get_test_seq(torch.FloatTensor(img_x).cuda(), torch.FloatTensor(state).cuda())
#pred_action = self.policy(torch.FloatTensor(img_x).cuda()).detach().cpu().numpy()
        pred_action = self.actions[t].cpu().numpy()
        min_norm, max_norm = [-0.12025934, -0.12001299, -0.25635245, -3.5012    ], [0.17273517, 0.1317415 , 0.27066168, 3.3369567 ] 
        min_norm, max_norm = np.array(min_norm), np.array(max_norm)
        midpt = (min_norm + max_norm) / 2
        pred_action = pred_action * (max_norm - min_norm) / 2 + midpt 
        pred_action = np.squeeze(pred_action)
        print(pred_action)
        #pred_action *= 3.5 * np.array([0.02303233, 0.02453148, 0.03293544, 0.50120825])
#pred_action *= 5

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


def load_png_image(path):
    img_x = cv2.imread(path) 
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.
    img_x = cv2.resize(img_x, (64, 64), interpolation=cv2.INTER_AREA).flatten()[None]
    return img_x

if __name__ == '__main__':
    p = PolicyNet(False)
    p.load_state_dict(torch.load('/home/stephen/offline_rl_models/bc/bc_policy_recurr_noaugtest.pkl'))
    p.eval()
    p = p.cuda()

