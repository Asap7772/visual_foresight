import numpy as np
from visual_mpc.policy.policy import Policy
import os


class PControllerPolicy(Policy):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

    def _default_hparams(self):
        default_dict = {
            'log': False,
            'goal_pos': (0.5, 0.5, 0.5, 0.0, 0.0),
        }
        parent_params = super(PControllerPolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(PControllerPolicy, self).set_log_dir(d)

    def act(self, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None, verbose_worker=None):
        c_state = state[-1]
        action_vector = np.array(self._hp.goal_pos) - c_state
        action_vector[3:5] = 0
        action_vector = action_vector[:4]
        return {'actions': 0.1*action_vector/np.linalg.norm(action_vector)}

