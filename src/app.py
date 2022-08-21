import argparse
import itertools
import math
import numpy as np
import os
import sys
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from iterators import EpochBatchIterator
from models import SelfAttentiveCNFSpatiotemporalModel
from models.spatial import SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
import utils
from Trainer import Trainer


def args_set():
    parser = argparse.ArgumentParser()

    # parameters of generated data
    parser.add_argument("--data_num", type=int, default=100000)
    parser.add_argument("--sim_batch", type=int, default=10000)

    # network parameters
    parser.add_argument("--type_dim", type=int, default=8)
    parser.add_argument("--week_dim", type=int, default=1)
    parser.add_argument("--hour_dim", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--nflows", type=int, default=1)
    parser.add_argument("--nattn", type=int, default=4)
    parser.add_argument("--step_size", type=float, default=1e-2)
    parser.add_argument("--tpp", type=str, choices=["poisson", "hawkes", "correcting", "neural"], default="neural")
    parser.add_argument("--actfn", type=str, default="swish")
    parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
    parser.add_argument("--hdims", type=str, default="64-64-64")
    parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")
    parser.add_argument("--tpp_hdims", type=str, default="32-32")
    parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
    parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
    parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
    parser.add_argument("--solve_reverse", action="store_true")
    parser.add_argument("--l2_attn", action="store_false")
    parser.add_argument("--naive_hutch", action="store_true")
    parser.add_argument("--ode_method", type=str, default="scipy_solver",choices=["scipy_solver","dopri5","rk4","euler"])
    parser.add_argument("--ode_solver", type=str, default="RK45")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--otreg_strength", type=float, default=1e-4)
    parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)
    parser.add_argument("--max_intensity", type=float, default=20.0)

    # training
    # iteration
    parser.add_argument("--sim_iterations", type=int, default=5000)
    parser.add_argument("--iter_ppo", type=int, default=2)
    parser.add_argument("--iter_disc", type=int, default=8)
    parser.add_argument("--iter_enc", type=int, default=2)
    parser.add_argument("--disc_epoch", type=int, default=5)
    parser.add_argument("--ppo_epoch", type=int, default=10)
    parser.add_argument("--enc_epoch", type=int, default=5)
    parser.add_argument("--value_epoch", type=int, default=10)
    parser.add_argument("--simfreq", type=int, default=5)
    parser.add_argument("--pretrain_epoch", type=int, default=20)
    # hyper parameter
    parser.add_argument('--gamma', type=float, default=0.98, help='None')
    parser.add_argument('--lmbda', type=float, default=0.95, help='None')
    parser.add_argument('--entropy_coef', type=float, default=1e-2, help='None')
    parser.add_argument('--entropy_decay', type=float, default=0.5, help='None')
    parser.add_argument('--adv_norm', type=int, default=1, help='None')
    # clip
    parser.add_argument('--clip_grad', type=float, default=1e4, help='None')
    parser.add_argument('--clip_param', type=float, default=0.25, help='None')
    parser.add_argument('--disc_clip', type=int, default=1, help='None')
    parser.add_argument('--max_ratio', type=float, default=10.0, help='None')
    # lr
    parser.add_argument('--policy_lr', type=float, default=3e-5, help='None')
    parser.add_argument('--value_lr', type=float, default=1e-2, help='None')
    parser.add_argument('--dis_lr', type=float, default=1e-3, help='None')
    parser.add_argument('--enc_lr', type=float, default=3e-4, help='None')
    # batch 
    parser.add_argument("--enc_batch", type=int, default=64)
    parser.add_argument("--warmup_itrs", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=50000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradclip", type=float, default=50)


    # cuda
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--is_cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)


    return parser.parse_args()

args = args_set()

def initial_setting():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

initial_setting()

if __name__ == '__main__':

    device = torch.device("cuda:{}".format(args.cuda_id) if args.is_cuda else "cpu")

    TIME = int(time.time())
    TIME = time.localtime(TIME)
    TIME = time.strftime("%Y-%m-%d %H:%M:%S",TIME)

    Trainer = Trainer_GAIL(args, device, TIME)
    Trainer.Train()

