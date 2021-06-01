from multiprocessing import Pool, Process, Manager
import argparse
import importlib.machinery
import importlib.util
import pdb
import copy
import random
import numpy as np
import re
import os
from visual_mpc.sim.util.combine_score import combine_scores
import ray
from visual_mpc.sim.simulator import Sim
from visual_mpc.sim.benchmarks import run_trajectories
from visual_mpc.utils.sync import SyncCounter

def single_traj_worker(conf, iex=-1, ngpu=1):
    print('started process with PID:', os.getpid())
    print('making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    ))

    random.seed(None)
    np.random.seed(None)
    s = Sim(conf)
    s.run()


def bench_worker(conf, iex=-1, ngpu=1):
    print('started process with PID:', os.getpid())
    random.seed(None)
    np.random.seed(None)
    print('start ind', conf['start_index'])
    print('end ind', conf['end_index'])
    run_trajectories(conf, iex, gpu_id=conf['gpu_id'], ngpu=ngpu)

class ControlManager:
    def __init__(self, save_dir_prefix='', args_in=None, hyperparams=None):
        """
        :param save_dir_prefix: will be added to the experiment data and training data save paths
        specified in the variables $VMPC_DATA and $VMPC_EXP
        :param args_in:
        :param hyperparams:
        """
        parser = argparse.ArgumentParser(description='run parallel data collection')
        parser.add_argument('experiment', type=str, help='path to the experiment configuraiton file including mod_hyper.py')
        parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
        parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
        parser.add_argument('--ngpu', type=int, help='the number of gpus to use', default=1)
        parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
        parser.add_argument('--isplit', type=int, help='split id', default=-1)
        parser.add_argument('--cloud', dest='cloud', action='store_true', default=False)
        parser.add_argument('--ray', dest='use_ray', action='store_true', default=False)
        parser.add_argument('--save_thread', dest='save_thread', action='store_true', default=False)
        parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)

        args = parser.parse_args(args_in)
    
        if hyperparams is None:
            hyperparams_file = args.experiment
            loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            self.hyperparams = mod.config
        else:
            self.hyperparams = hyperparams
        self.args = args
        self.save_dir_prefix = save_dir_prefix

    def run(self, logging_conf=None):
        args = self.args
        hyperparams = self.hyperparams
    
        gpu_id = args.gpu_id
        n_worker = args.nworkers
        if args.nworkers == 1:
            parallel = False
        else:
            parallel = True
        print('parallel ', bool(parallel))

        end_idx, n_traj, start_idx = self.get_startend_idx(args, hyperparams, n_worker)

        if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
            try:
                os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
            except: pass

        self.set_paths(hyperparams)
    
        if args.save_thread:
            record_queue, record_saver_proc, counter = prepare_saver(hyperparams)
        else: record_queue, record_saver_proc, counter = None, None, None

        if args.iex != -1:
            hyperparams['agent']['iex'] = args.iex
    
        conflist = []
        for i in range(n_worker):
            modconf = copy.deepcopy(hyperparams)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]
            modconf['ntraj'] = n_traj
            modconf['gpu_id'] = i + gpu_id
            if args.save_thread:
                modconf['record_saver'] = record_queue
                modconf['counter'] = counter
                modconf['use_save_thread'] = True
            if logging_conf is not None:
                modconf['logging_conf'] = logging_conf
            conflist.append(modconf)
        if parallel:
            if args.use_ray:
                self.ray_start_parallel(conflist, n_worker)
            else:
                self.start_parallel(conflist, n_worker)
        else:
            bench_worker(conflist[0], args.iex, args.ngpu)
    
        if args.save_thread:
            record_queue.put(None)           # send flag to background thread that it can end saving after it's done
            record_saver_proc.join()         # joins thread and continues execution
    
        if 'master_datadir' in hyperparams['agent']:
            ray.wait([sync_todo_id])
        self.scores = combine_scores(hyperparams, hyperparams['log_dir'])

    def get_startend_idx(self, args, hyperparams, n_worker):
        if args.nsplit != -1:
            assert args.isplit >= 0 and args.isplit < args.nsplit, "isplit should be in [0, nsplit-1]"

            n_persplit = max((hyperparams['end_index'] + 1 - hyperparams['start_index']) / args.nsplit, 1)
            hyperparams['end_index'] = int((args.isplit + 1) * n_persplit + hyperparams['start_index'] - 1)
            hyperparams['start_index'] = int(args.isplit * n_persplit + hyperparams['start_index'])
        n_traj = hyperparams['end_index'] - hyperparams['start_index'] + 1
        traj_per_worker = int(n_traj // np.float32(n_worker))
        start_idx = [hyperparams['start_index'] + traj_per_worker * i for i in range(n_worker)]
        end_idx = [hyperparams['start_index'] + traj_per_worker * (i + 1) - 1 for i in range(n_worker)]
        return end_idx, n_traj, start_idx

    def set_paths(self, hyperparams):
        """
        set two directories:
            log_dir is for experiment logs, visuals, tensorboards stuff etc.
            data_save_dir is for collected datasets
            the subpath after the experiments folder is appended to the $VMPC_DATA and $VMPC_EXP directories respectively
        """
        assert 'experiments' in self.args.experiment
        subpath = hyperparams['current_dir'].partition('experiments')[2]
        hyperparams['data_save_dir'] = os.environ['VMPC_DATA'] + '/' + self.save_dir_prefix + subpath
        hyperparams['log_dir'] = os.environ['VMPC_EXP'] + '/' + self.save_dir_prefix + subpath
        print('setting data_save_dir to', hyperparams['data_save_dir'])
        print('setting log_dir to', hyperparams['log_dir'])

    def start_parallel(self, conflist, n_worker):
        # mp.set_start_method('spawn')  # this is important for parallelism with xvfb
        p = Pool(n_worker)
        p.map(bench_worker, conflist)

    def ray_start_parallel(self, conflist, n_worker):
        head_node_address = ray.services.get_node_ip_address() + ':6379'
        ray.init(redis_address=head_node_address)
        # ray.init(local_mode=True)   # serial execution for debug
        # ray.init()   # use this without cluster
        ids = []
        for i in range(n_worker):
            ids.append(ray_bench_worker.remote(conflist[i]))
        [ray.get(id) for id in ids]


def prepare_saver(hyperparams):
    m = Manager()
    record_queue, synch_counter = m.Queue(), SynchCounter(m)
    save_dir, T = hyperparams['agent']['data_save_dir'] + '/records', hyperparams['agent']['T']
    if hyperparams.get('save_data', True) and not hyperparams.get('save_raw_images', False):
        seperate_good, traj_per_file = hyperparams.get('seperate_good', False), hyperparams.get('traj_per_file', 16)

        if 'save_format' in hyperparams:
            save_format = hyperparams['save_format']
        else: save_format = 'tfrec'

        # ToDO comment back in
        # record_saver_proc = Process(target=record_worker, args=(
        # record_queue, save_dir, T, seperate_good, traj_per_file, save_format, hyperparams['start_index']))
        # record_saver_proc.start()
        record_saver_proc = None # Todo: remove this!
    else:
        record_saver_proc = None
    return record_queue, record_saver_proc, synch_counter


def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    ControlManager().run()
