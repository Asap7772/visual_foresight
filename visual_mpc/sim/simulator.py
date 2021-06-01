import os
import os.path
from visual_mpc.agent.utils.utils import timed
import sys
from visual_mpc.agent.utils.raw_saver import RawSaver
from visual_mpc.agent.utils.traj_saver import GeneralAgentSaver
from visual_mpc.agent.utils.hdf5_saver import HDF5Saver
from tensorflow.contrib.training import HParams
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))



class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, gpu_id=0, ngpu=1):
        self._hp = self._default_hparams()
        self.override_defaults(config)
        self._hp.agent['log_dir'] = self._hp.log_dir
        self.agent = self._hp.agent['type'](self._hp.agent)
        self.policy = self._hp.policy['type'](self.agent._hp.values(), self._hp.policy, gpu_id, ngpu)

        self._record_queue = self._hp.record_saver
        self._counter = self._hp.counter

        self.trajectory_list = []
        self.im_score_list = []
        try:
            os.remove(self._hp.agent['image_dir'])
        except:
            pass

        self.savers = []

        if 'hdf5' in self._hp.save_format:
            self.savers.append(HDF5Saver(self._hp.data_save_dir, self.agent.env._hp, self.agent._hp,
                                     traj_per_file=self._hp.traj_per_file, offset=self._hp.start_index,
                                         split_train_val_test=self._hp.split_train_val_test))
        if 'raw' in self._hp.save_format:
            self.savers.append(RawSaver(self._hp.data_save_dir))
        if 'tfrec' in self._hp.save_format:
            self.savers.append(GeneralAgentSaver(self._hp.data_save_dir, self.agent.T, False,
                                     traj_per_file=self._hp.traj_per_file, offset=self._hp.start_index))

    def override_defaults(self, config):
        """
        :param config:  override default valus with config dict
        :return:
        """
        for name, value in config.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute {} is identical to default value!!".format(name))
            if name in self._hp and self._hp.get(name) is None:   # don't do a type check for None default values
                setattr(self._hp, name, value)
            else: self._hp.set_hparam(name, value)

    def _default_hparams(self):
        default_dict = {
            'save_format': ['hdf5', 'raw'],
            'save_data': True,
            'agent': {},
            'policy': {},
            'start_index': -1,
            'end_index': -1,
            'ntraj': -1,
            'gpu_id': -1,
            'current_dir': '',
            'record_saver': None,
            'counter': None,
            'traj_per_file': 10,
            'data_save_dir': '',
            'log_dir': '',
            'result_dir': '',
            'split_train_val_test': True,
            'logging_conf': None,   # only needed for training loop
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def run(self):
        if self._counter is None:
            for i in range(self._hp.start_index, self._hp.end_index+1):
                self.take_sample(i)
        else:
            itr = self._counter.ret_increment()
            while itr < self._hp.ntraj:
                print('taking sample {} of {}'.format(itr, self._hp.ntraj))
                self.take_sample(itr)
                itr = self._counter.ret_increment()
        self.agent.cleanup()

    @timed('traj sample time: ')
    def take_sample(self, index):
        """
        :param index:  run a single trajectory with index
        :return:
        """
        agent_data, obs_dict, policy_out = self.agent.sample(self.policy, index)
        if self._hp.save_data:
            self.save_data(index, agent_data, obs_dict, policy_out)

        return agent_data

    @timed('savingtime: ')
    def save_data(self, itr, agent_data, obs_dict, policy_outputs):
        if self._record_queue is not None:  # if using a queue to save data
            self._record_queue.put((agent_data, obs_dict, policy_outputs))
        else:
            for saver in self.savers: # if directly saving data
                saver.save_traj(itr, agent_data, obs_dict, policy_outputs)
