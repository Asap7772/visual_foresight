import ipdb
import numpy as np
import cv2
import os
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_mpc.policy.policy import Policy


class CoordConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding=1, stride=1):
        super(CoordConv, self).__init__()
        self._conv = nn.Conv2d(in_dim + 2, out_dim, kernel_size, stride, padding)

    def forward(self, x):
        B, C, H, W = x.shape
        h_pad = torch.linspace(-1, 1, H).reshape((1, 1, H, 1)).repeat((B, 1, 1, W))
        w_pad = torch.linspace(-1, 1, W).reshape((1, 1, 1, W)).repeat((B, 1, H, 1))
        x = torch.cat((x, h_pad.to(x.device), w_pad.to(x.device)), 1)
        return self._conv(x)


class VGGFeats(nn.Module):
    def __init__(self):
        super(VGGFeats, self).__init__()
        vgg_feats = models.vgg16(pretrained=True).features
        vgg_feats = list(vgg_feats.children())[:4]

        cc0 = CoordConv(64, 64, 3, stride=2)
        n0 = nn.InstanceNorm2d(64, affine=True)
        a0 = nn.ReLU(inplace=True)

        cc1 = CoordConv(64, 128, 3, stride=2)
        n1 = nn.InstanceNorm2d(128, affine=True)
        a1 = nn.ReLU(inplace=True)
        p1 = nn.MaxPool2d(2)

        cc2_1 = CoordConv(128, 256, 3)
        n2_1 = nn.InstanceNorm2d(256, affine=True)
        a2_1 = nn.ReLU(inplace=True)
        cc2_2 = CoordConv(256, 256, 3)
        n2_2 = nn.InstanceNorm2d(256, affine=True)
        a2_2 = nn.ReLU(inplace=True)
        p2 = nn.AdaptiveAvgPool2d(1)

        visual_feats = vgg_feats + [cc0, n0, a0, cc1, n1, a1, p1, cc2_1, n2_1, a2_1, cc2_2, n2_2, a2_2, p2]
        self._v = nn.Sequential(*visual_feats)

        self._fc1 = nn.Linear(256, 32)
        self._a1 = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(100, 16)
        self._a2 = nn.ReLU(inplace=True)
        self._fc3 = nn.Linear(16, 4)

    def forward(self, imgs, state):
        B, T, C, H, W = imgs.shape
        vis = self._v(imgs.reshape((B * T, C, H, W)))
        vis = self._a1(self._fc1(vis.reshape((B, T, 256))))
        vis = vis.reshape((B, 96))
        vis_state = torch.cat((vis, state), 1)
        return self._fc3(self._a2(self._fc2(vis_state)))


class MultiViewReachPolicy(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.device = 'cuda:{}'.format(gpu_id)
        self.load_model(gpu_id, ngpu)
        self.image_height, self.image_width = ag_params['image_height'], ag_params['image_width']
        # image normalization
        self._mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).to(self.device)
        self._std = torch.FloatTensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).to(self.device)
        self.norm_transform = transforms.Normalize(self._mean, self._std)

    def load_model(self, gpu_id, ngpu):
        self.model = VGGFeats()
        self.model = self.model.to(self.device).eval()
        print('Loading model from {}'.format(self._hp.path))
        self.model.load_state_dict(torch.load(self._hp.path))

    def _default_hparams(self):
        default_dict = {
            'user': False,
            'path': '/home/asap7772/batch_rl_private/data/lagrange-10-robonet-widowx/302109/lagrange_10_robonet_widowx/302109_2020_05_27_01_42_24_0000--s-0',
            'goal_dir': '',
            'policy_type': 2,
            'goal_pos': (0.935024, 0.204873, 0.0694792),
            'log': False,
            'num_views': 1,
        }
        parent_params = super(MultiViewReachPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(MultiViewReachPolicy, self).set_log_dir(d)

    def load_goals(self):
        goal_dir = self._hp.goal_dir
        goal_imgs = []
        for i in range(2):
            im_pillow = np.array(Image.open('{}/goal_{}.jpg'.format(goal_dir, i)))
            im_bgr = cv2.resize(im_pillow, (self.image_width, self.image_height), cv2.INTER_AREA)
            im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2BGR)
            im = transforms.ToTensor()(im_bgr).to(self.device)
            im = self.norm_transform(im)
            goal_imgs.append(im)
        goal_imgs = torch.stack(goal_imgs, axis=0)
        # [num_goals, C, H, W]
        return goal_imgs[None]

    def preprocess_img(self, img):
        img -= self._mean
        img /= self._std
        return img

    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, verbose_worker=None):
        goal_images = self.load_goals()
        # goal images should be in format [1, #_goals, C, W, H]

        with torch.no_grad():
            recent_img = images[-1]
            resized = []
            stacked_images = [recent_img[2] for i in range(self._hp.num_views)]
            for img in stacked_images:
                img = transforms.ToTensor()(img).to(self.device)
                img = self.norm_transform(img)
                assert img.shape[1] == self.image_height, img.shape[2] == self.image_width
                resized.append(img)
            img_x = torch.cat(resized, dim=0)
            # [num_views * 3, h, w]

            img_in = torch.cat((img_x[None, None], goal_images), 1).to(self.device)
            state_in = torch.FloatTensor(state[-1][None][:, :4]).to(self.device)

            action = self.model(img_in, state_in)
            action = torch.squeeze(action).detach().cpu().numpy()

        return {
            'actions': action,
        }
